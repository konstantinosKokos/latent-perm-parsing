from itertools import chain
from typing import *

import torch
from torch import Tensor, LongTensor
from torch.nn import Module, Dropout
from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from transformers import RobertaTokenizer

from Parser.data.sample import Sample, strs, ints, Matrix


class Tokenizer(object):
    def __init__(self):
        self.core = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")

    def encode_sample(self, sample: Sample) -> ints:
        return self.core.encode(' '.join(sample.words))

    def encode_samples(self, samples: List[Sample]) -> List[ints]:
        return [self.encode_sample(sample) for sample in samples]

    def encode_sent(self, sent: str) -> ints:
        return self.core.encode(sent)

    def encode_sents(self, sents: strs) -> List[ints]:
        return [self.encode_sent(sent) for sent in sents]


def pad_sequence(x, **kwargs):
    return _pad_sequence(x, batch_first=True, **kwargs)


def make_sinkhorn_inputs(bsd_tensor: Tensor, positional_ids: List[List[LongTensor]], device: str = 'cpu') \
        -> List[List[Tensor]]:
    """

    :param bsd_tensor:
        Tensor of shape batch size \times sequence length \times feature dimensionality.
    :param positional_ids:
        A List of batch_size elements, each being a List of num_atoms LongTensors.
        Each LongTensor in positional_ids[b][a] indexes the location of atoms of type a in sentence b.
    :param device:
    :return:
    """

    return [[bsd_tensor.select(0, index=i).index_select(0, index=atom.to(device)) for atom in sentence]
            for i, sentence in enumerate(positional_ids)]


def tensorize_sentence_indexers(indexers: List[ints]) -> List[LongTensor]:
    return [LongTensor(index) for index in indexers]


def tensorize_batch_indexers(sentences: List[List[ints]]) -> List[List[LongTensor]]:
    return [tensorize_sentence_indexers(sentence) for sentence in sentences]


def sents_to_batch(sents: strs, tokenizer: Tokenizer) -> LongTensor:
    tokens = tokenizer.encode_sents(sents)
    return pad_sequence([LongTensor(token_seq) for token_seq in tokens], padding_value=tokenizer.core.pad_token_id)


class AtomTokenizer(object):
    def __init__(self, atom_map: Mapping[str, int]):
        self.atom_map = atom_map
        self.inverse_atom_map = {v: k for k, v in self.atom_map.items()}
        self.sep_token = '[SEP]'
        self.sos_token = '[SOS]'
        self.pad_token = '[PAD]'
        self.sep_token_id = self.atom_map[self.sep_token]
        self.sos_token_id = self.atom_map[self.sos_token]
        self.pad_token_id = self.atom_map[self.pad_token]

    def __len__(self) -> int:
        return len(self.atom_map)

    def convert_atoms_to_ids(self, atoms: Iterable[str]) -> ints:
        return [self.atom_map[str(atom)] for atom in atoms]

    def convert_sents_to_ids(self, sentences: Iterable[Iterable[str]]) -> List[ints]:
        return [self.convert_atoms_to_ids(atoms) for atoms in sentences]

    def convert_ids_to_atoms(self, ids: Iterable[int]) -> strs:
        return [self.inverse_atom_map[int(i)] for i in ids]

    def convert_batch_ids_to_atoms(self, batch_ids: Iterable[Iterable[int]],
                                   max_lens: Optional[ints] = None) -> List[Optional[strs]]:
        if max_lens is not None:
            return [self.drop_padded_atoms(self.convert_ids_to_atoms(ids), length)
                    for ids, length in zip(batch_ids, max_lens)]
        return [self.convert_ids_to_atoms(ids) for ids in batch_ids]

    def convert_batch_ids_to_polish(self, batch_ids: List[List[int]],
                                    max_lens: ints) -> List[Optional[List[strs]]]:
        sents = [self.drop_padded_atoms(self.convert_ids_to_atoms(ids), length)
                 for ids, length in zip(batch_ids, max_lens)]
        return [None if sent is None else [t.split() for t in sent] for sent in sents]

    def convert_beam_ids_to_polish(self, beam_ids: List[List[ints]], max_lens: ints) \
            -> List[List[Optional[List[strs]]]]:
        return [self.convert_batch_ids_to_polish(sent, [length] * len(sent))
                for sent, length in zip(beam_ids, max_lens)]

    @staticmethod
    def drop_padded_atoms(atoms: Iterable[str], max_len: int, sep: str = '[SEP]') -> Optional[strs]:
        if len(list(filter(lambda x: x == sep, atoms))) < max_len:
            return None
        ret = ' ' + ' '.join(atoms[1:])
        ret = ret.split(f'{sep}')[:max_len]
        return [r[1:-1] for r in ret]

    def samples_to_batch(self, samples: List[Sample], tokenizer: Tokenizer) \
            -> Tuple[LongTensor, LongTensor, List[List[LongTensor]], List[List[LongTensor]]]:
        return samples_to_batch(samples, self, tokenizer)


Item = Tuple[LongTensor, LongTensor, List[LongTensor], List[LongTensor], List[Matrix]]
Batch = Tuple[LongTensor, LongTensor, List[List[LongTensor]], List[List[LongTensor]], List[LongTensor]]


def vectorize_sample(sample: Sample, atokenizer: 'AtomTokenizer', tokenizer: Tokenizer) -> Item:
    symbols = tokenizer.encode_sample(sample)
    embedding_ids = LongTensor(symbols)
    polishes = sample.polish
    _atom_ids = atokenizer.convert_atoms_to_ids(polishes)
    atom_ids = LongTensor(_atom_ids)
    _positives = sample.positive_ids
    _negatives = sample.negative_ids
    positives = tensorize_sentence_indexers(_positives)
    negatives = tensorize_sentence_indexers(_negatives)
    matrices = sample.matrices
    return embedding_ids, atom_ids, positives, negatives, matrices


def batchify_vectorized_samples(inps: List[Item], padding_value_word: int, padding_value_symbol: int,
                                max_difficulty: int, exclude_singular: bool) -> Batch:

    _embedding_ids, _atom_ids, _positives, _negatives, _matrices = list(zip(*inps))
    embedding_ids = pad_sequence(_embedding_ids, padding_value=padding_value_word)
    atom_ids = pad_sequence(_atom_ids, padding_value=padding_value_symbol)
    return embedding_ids, atom_ids, _positives, _negatives, make_permutors(_matrices, max_difficulty, exclude_singular)


def samples_to_batch(samples: List[Sample], atokenizer: 'AtomTokenizer', tokenizer: Tokenizer) \
        -> Tuple[LongTensor, LongTensor, List[List[LongTensor]], List[List[LongTensor]]]:
    symbols: List[ints] = tokenizer.encode_samples(samples)
    embedding_ids: LongTensor = pad_sequence([LongTensor(sample) for sample in symbols],
                                             padding_value=tokenizer.core.pad_token_id)

    polishes: List[strs] = [sample.polish for sample in samples]
    _atom_ids: List[ints] = atokenizer.convert_sents_to_ids(polishes)
    atom_ids: LongTensor = pad_sequence([LongTensor(sample) for sample in _atom_ids])

    _positives: List[List[ints]] = [sample.positive_ids for sample in samples]
    _negatives: List[List[ints]] = [sample.negative_ids for sample in samples]

    positives = tensorize_batch_indexers(_positives)
    negatives = tensorize_batch_indexers(_negatives)
    return embedding_ids, atom_ids, positives, negatives


def make_atom_mapping(samples: List[Sample]) -> Mapping[str, int]:
    polishes: strs = list(chain.from_iterable(map(lambda sample: sample.polish, samples)))
    return {**{'[PAD]': 0}, **{p: i + 1 for i, p in enumerate(sorted(set(polishes)))}}


def measure_linking_accuracy(pred: List[Tensor], truth: List[Tensor]) -> Tuple[int, int]:
    def measure_one(pred_one: Tensor, truth_one: LongTensor) -> Tuple[int, int]:
        b, n = pred_one.shape[0:2]
        return torch.sum(pred_one.argmax(dim=-1) == truth_one).item(), b * n

    measurements = list(map(measure_one, pred, truth))
    correct, total = list(zip(*measurements))

    return sum(correct), sum(total)


def measure_supertagging_accuracy(pred: LongTensor, truth: LongTensor, ignore_idx: int = 0) \
        -> Tuple[Tuple[int, int], Tuple[int, int]]:
    correct_words = torch.ones(pred.size())
    correct_words[pred != truth] = 0
    correct_words[truth == ignore_idx] = 1

    correct_sentences = correct_words.prod(dim=1)
    num_correct_sentences = correct_sentences.sum().item()

    num_correct_words = correct_words.sum().item()
    num_masked_words = len(truth[truth == ignore_idx])

    return (num_correct_sentences, pred.shape[0]), \
           (num_correct_words - num_masked_words, pred.shape[0] * pred.shape[1] - num_masked_words)


def make_permutors(matrices: List[List[Matrix]], max_difficulty: int, exclude_singular: bool = True) \
        -> List[LongTensor]:
    def tensorize_matrix(matrix: Matrix) -> LongTensor:
        return LongTensor(matrix).argmax(dim=-1)

    permutors = list(filter(lambda matrix: matrix, chain.from_iterable(matrices)))
    distinct_shapes = set(map(lambda permutor: len(permutor), permutors))
    if exclude_singular:
        distinct_shapes = distinct_shapes.difference({1})
    distinct_shapes = sorted(distinct_shapes)

    grouped_permutors = []
    for shape in distinct_shapes:
        if shape > max_difficulty:
            break
        this_shape_permutors = list(filter(lambda permutor: len(permutor) == shape, permutors))
        grouped_permutors.append(torch.stack(list(map(lambda permutor:
                                                      tensorize_matrix(permutor),
                                                      this_shape_permutors))))

    return grouped_permutors


class FuzzyLoss(Module):
    def __init__(self, loss_fn: Module, num_classes: int,
                 mass_redistribution: float, ignore_index: List[int]) -> None:
        super(FuzzyLoss, self).__init__()
        self.loss_fn = loss_fn
        self.nc = num_classes
        self.mass_redistribution = mass_redistribution
        self.ignore_index = ignore_index

    def __call__(self, x: Tensor, y: LongTensor) -> Tensor:
        y = y.flatten()
        y_float = torch.zeros(x.shape[0] * x.shape[1], self.nc, device=x.device, dtype=torch.float)
        y_float.fill_(self.mass_redistribution / (self.nc-(1 + len(self.ignore_index))))
        y_float.scatter_(1, y.unsqueeze(1), 1 - self.mass_redistribution)
        mask = torch.zeros_like(y, dtype=torch.bool)
        for idx in self.ignore_index:
            mask = torch.bitwise_or(mask, y == idx)
        y_float[mask.unsqueeze(1).repeat(1, self.nc)] = 0
        return self.loss_fn(torch.log_softmax(x.view(-1, self.nc), dim=-1), y_float)


def count_parameters(model: Module) -> int:
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = torch.prod(torch.tensor(param.size()))
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param
