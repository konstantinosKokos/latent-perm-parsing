import os
import subprocess
import sys

from torch.nn import KLDivLoss

from .data.preprocessing import load_stored
from .neural.model import *
from .neural.schedules import *
from .neural.utils import *


decoder_epochs = 40
tag_epochs = 80
parse_epochs = 80


def logprint(x: str, ostreams: list[Any]) -> None:
    print(x)
    for ostream in ostreams:
        ostream.write(x + '\n')
    sys.stdout.flush()


def load_model(parser: Parser, load: str, **kwargs) -> tuple[int, Dict, int]:
    print('Loading model parameters...')
    temp = torch.load(load, **kwargs)
    parser.load_state_dict(temp['model_state_dict'], strict=False)
    step_num = temp['step']
    opt_state_dict = temp['opt_state_dict']
    epoch = temp['epoch']
    return step_num, opt_state_dict, epoch


def init(datapath: Optional[str] = None, max_len: int = 128, train_batch: int = 32,
         val_batch: int = 128, device: str = 'cuda', version: Optional[str] = None,
         save_to_dir: Optional[str] = None) \
        -> tuple[DataLoader, DataLoader, DataLoader, int, Parser, str]:
    if version is None:
        version = subprocess.check_output(['git', 'describe', '--always'], cwd='./Parser').strip().decode()
    if save_to_dir is not None:
        os.makedirs(f'{save_to_dir}/', exist_ok=True)
    else:
        os.makedirs(f'./stored_models/{version}/', exist_ok=True)
    print(f'Version id:\t{version}')

    # load data and model
    _trainset, _devset, _testset = load_stored() if datapath is None else load_stored(datapath)

    print('Making tokenizers...')
    atom_map = make_atom_mapping(_trainset+_devset+_testset)
    atokenizer = AtomTokenizer(atom_map)
    tokenizer = Tokenizer()

    print('Making dataloaders...')
    trainset, devset, testset = list(map(lambda dset: list(filter(lambda sample: len(sample.polish) <= max_len, dset)),
                                         [_trainset, _devset, _testset]))
    nbatches = get_nbatches(trainset, train_batch)
    trainset = [vectorize_sample(s, atokenizer, tokenizer) for s in trainset]
    devset = [vectorize_sample(s, atokenizer, tokenizer) for s in sorted(devset, key=lambda x: len(x.polish))]
    testset = [vectorize_sample(s, atokenizer, tokenizer) for s in sorted(testset, key=lambda x: len(x.polish))]

    train_dl = make_dataloader(trainset, atokenizer.pad_token_id, tokenizer.core.pad_token_id, 32, True,
                               train_batch, True)
    dev_dl = make_dataloader(devset, atokenizer.pad_token_id, tokenizer.core.pad_token_id, 32, False, val_batch, False)
    test_dl = make_dataloader(testset, atokenizer.pad_token_id, tokenizer.core.pad_token_id, 32, False,
                              val_batch, False)

    print('Initializing model...')
    parser = Parser(atokenizer, tokenizer, device=device)

    return train_dl, dev_dl, test_dl, nbatches, parser, version


def init_without_datasets(atom_map_path: str = './Parser/data/atom_map.txt', device: str = 'cuda') -> Parser:
    with open(atom_map_path, 'r') as f:
        atom_map = dict(map(lambda pair: (pair[0], int(pair[1])), map(lambda line: line.split('\t'), f.readlines())))
    atokenizer = AtomTokenizer(atom_map)
    tokenizer = Tokenizer()
    return Parser(atokenizer, tokenizer, device=device)


def train(model_path: Optional[str] = None, data_path: Optional[str] = None,
          version: Optional[str] = None, save_to_dir: Optional[str] = None):

    train_dl, val_dl, test_dl, nbatches, parser, version = init(data_path, version=version, save_to_dir=save_to_dir)

    _opt = torch.optim.AdamW(parser.parameters(), lr=1e10, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-02)
    dec_schedule = make_cosine_schedule(max_lr=5e-04, warmup_steps=1000, decay_over=decoder_epochs * nbatches)
    tag_schedule = make_cosine_schedule(max_lr=1e-04, warmup_steps=5000, decay_over=tag_epochs * nbatches)
    parse_schedule = make_cosine_schedule(max_lr=5e-05, warmup_steps=1000, decay_over=parse_epochs * nbatches)

    fuzzy_loss = FuzzyLoss(KLDivLoss(reduction='batchmean'), len(parser.atom_tokenizer), 0.1,
                           ignore_index=[parser.atom_tokenizer.pad_token_id, parser.atom_tokenizer.sos_token_id])

    if model_path is not None:
        print('Loading checkpoint...')
        step_num, opt_dict, init_epoch = load_model(parser, model_path)
        opt = Scheduler(_opt, (dec_schedule if init_epoch < decoder_epochs
                               else (tag_schedule if init_epoch < decoder_epochs + tag_epochs
                                     else parse_schedule)))
        opt.step_num = step_num
        opt.lr = opt.schedule(opt.step_num)
        opt.opt.load_state_dict(opt_dict)
    else:
        opt = Scheduler(_opt, dec_schedule)
        init_epoch = 0

    if save_to_dir is None:
        save_to_dir = './stored_models'

    for e in range(init_epoch, decoder_epochs + tag_epochs + parse_epochs):
        validate = e % 5 == 0
        save = e % 10 == 0
        linking_weight = 0.5 if e >= decoder_epochs + tag_epochs else 0.

        if save and e != init_epoch:
            print('\tSaving')
            torch.save({'model_state_dict': parser.state_dict(),
                        'opt_state_dict': opt.opt.state_dict(),
                        'step': opt.step_num,
                        'epoch': e},
                       f'{save_to_dir}/{version}/{e}.model')

        if e < decoder_epochs:
            with open(f'{save_to_dir}/{version}/log.txt', 'a') as stream:
                logprint('=' * 64, [stream])
                logprint(f'Pre-epoch {e}', [stream])
                logprint(' ' * 50 + f'LR: {opt.lr}\t({opt.step_num})', [stream])
                pretrain_loss = parser.pretrain_decoder_epoch(train_dl, fuzzy_loss, opt)
                logprint(f' Pretrain loss:\t\t{pretrain_loss:5.2f}', [stream])
                logprint(f' Pretrain accu:\t\t{parser.preval_epoch(val_dl)}', [stream])
                continue
        elif e == decoder_epochs:
            opt = Scheduler(opt.opt, tag_schedule)
        elif e == decoder_epochs + tag_epochs:
            opt = Scheduler(opt.opt, parse_schedule)

        with open(f'{save_to_dir}/{version}/log.txt', 'a') as stream:
            logprint('=' * 64, [stream])
            logprint(f'Epoch {e}', [stream])
            logprint(' ' * 50 + f'LR: {opt.lr}\t({opt.step_num})', [stream])
            logprint(' ' * 50 + f'LW: {linking_weight}', [stream])
            logprint('-' * 64, [stream])
            supertagging_loss, linking_loss = parser.train_epoch(train_dl, fuzzy_loss, opt, linking_weight)
            logprint(f' Supertagging Loss:\t\t{supertagging_loss:5.2f}', [stream])
            logprint(f' Linking Loss:\t\t\t{linking_loss:5.2f}', [stream])
            if validate:
                with open(f'{save_to_dir}/{version}/val_log.txt', 'a') as valstream:
                    logprint('-' * 64, [stream, valstream])
                    sentence_ac, atom_ac, link_ac = parser.eval_epoch(val_dl, link=linking_weight != 0)
                    logprint(f' Sentence Accuracy:\t\t{(sentence_ac * 100):6.2f}', [stream, valstream])
                    logprint(f' Atom Accuracy:\t\t{(atom_ac * 100):6.2f}', [stream, valstream])
                    logprint(f' Link Accuracy:\t\t{(link_ac * 100):6.2f}', [stream, valstream])
            logprint('\n', [stream])

