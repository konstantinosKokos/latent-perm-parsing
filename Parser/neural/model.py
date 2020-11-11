from torch.nn import functional, LayerNorm, Sequential
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaModel

from Parser.neural.sinkhorn import sinkhorn_fn_no_exp as sinkhorn
from Parser.neural.utils import *
from Parser.neural.embedding import ComplexEmbedding
from Parser.parsing.utils import TypeParser, Analysis
from Parser.neural.transformer import make_decoder, FFN


class Parser(Module):
    def __init__(self, atokenizer: AtomTokenizer, tokenizer: Tokenizer, enc_dim: int = 768,
                 dec_dim: int = 256, device: str = 'cpu'):
        super(Parser, self).__init__()
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.num_embeddings = len(atokenizer)
        self.device = device
        self.atom_tokenizer = atokenizer
        self.type_parser = TypeParser(atokenizer)
        self.tokenizer = tokenizer
        self.dropout = Dropout(0.1)
        self.enc_heads = 8
        self.dec_heads = 8
        self.d_atn_dec = self.dec_dim//self.dec_heads

        self.word_encoder = RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base").to(device)
        self.supertagger = make_decoder(num_layers=3, num_heads_enc=self.enc_heads, num_heads_dec=self.dec_heads,
                                        d_encoder=self.enc_dim, d_decoder=self.dec_dim,
                                        d_atn_enc=self.enc_dim//self.enc_heads, d_atn_dec=self.d_atn_dec,
                                        d_v_enc=self.enc_dim//self.enc_heads, d_v_dec=self.dec_dim//self.dec_heads,
                                        d_interm=self.dec_dim * 2, dropout_rate=0.1).to(device)
        self.atom_embedder = ComplexEmbedding(self.num_embeddings, self.dec_dim//2).to(device)
        self.linker = make_decoder(num_layers=1, num_heads_enc=self.enc_heads, num_heads_dec=self.dec_heads,
                                   d_encoder=self.enc_dim, d_decoder=self.dec_dim,
                                   d_atn_enc=self.enc_dim//self.enc_heads, d_atn_dec=self.d_atn_dec,
                                   d_v_enc=self.enc_dim//self.enc_heads, d_v_dec=self.dec_dim//self.dec_heads,
                                   d_interm=self.dec_dim * 2, dropout_rate=0.1).to(device)
        self.pos_transformation = Sequential(
            FFN(d_model=self.dec_dim, d_ff=self.dec_dim, d_out=32),
            LayerNorm(32)).to(device)
        self.neg_transformation = Sequential(
            FFN(d_model=self.dec_dim, d_ff=self.dec_dim, d_out=32),
            LayerNorm(32)).to(device)

        self.freeze_encoder()
        self.unfrozen_blocks = set(range(12))
        for block in self.unfrozen_blocks:
            self.unfreeze_encoder_block(block)

    def forward(self, *args) -> NoReturn:
        raise NotImplementedError('Forward not implemented.')

    @staticmethod
    def sinkhorn(x: Tensor, iters: int, tau: int = 1) -> Tensor:
        return sinkhorn(x, tau=tau, iters=iters)

    def freeze_encoder(self) -> None:
        for param in self.word_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder_block(self, block: int) -> None:
        dict(self.word_encoder.named_modules())[f'encoder.layer.{block}'].train()
        for param in dict(self.word_encoder.named_modules())[f'encoder.layer.{block}'].parameters():
            param.requires_grad = True

    def train(self, mode: bool = True) -> None:
        self.atom_embedder.train(mode)
        self.supertagger.train(mode)
        self.linker.train(mode)
        self.dropout.train(mode)
        self.pos_transformation.train(mode)
        self.neg_transformation.train(mode)
        for block in self.unfrozen_blocks:
            dict(self.word_encoder.named_modules())[f'encoder.layer.{block}'].train(mode)

    def eval(self) -> None:
        self.train(False)

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = torch.FloatTensor) -> 'Parser':
        self.device = self.device if device is None else device
        return super(Parser, self).to(device=device, dtype=dtype)

    def make_mask(self, inps: LongTensor, padding_id: int) -> LongTensor:
        mask = torch.ones_like(inps)
        mask[inps == padding_id] = 0
        return mask.to(self.device)

    def make_word_mask(self, lexical_ids: LongTensor) -> LongTensor:
        return self.make_mask(lexical_ids, self.tokenizer.core.pad_token_id)

    def make_atom_mask(self, atom_ids: LongTensor) -> LongTensor:
        return self.make_mask(atom_ids, self.atom_tokenizer.pad_token_id).unsqueeze(1).repeat(1, atom_ids.shape[1], 1)

    def make_atom_mask_from_lens(self, lens: ints) -> LongTensor:
        ones = torch.ones((len(lens), max(lens), max(lens)))
        for i, l in enumerate(lens):
            ones[i, :, l:] = 0
        return ones.to(self.device)

    def make_decoder_mask(self, b: int, n: int) -> LongTensor:
        upper_triangular = torch.triu(torch.ones(b, n, n), diagonal=1)
        return (torch.ones(b, n, n) - upper_triangular).to(self.device)

    def extend_mask(self, mask: LongTensor, size: int) -> LongTensor:
        return mask.unsqueeze(1).repeat(1, size, 1)

    def precode(self, lexical_token_ids: LongTensor, symbol_ids: LongTensor) -> \
            Tuple[Tensor, Tensor, LongTensor, LongTensor]:
        b, s_in = lexical_token_ids.shape
        s_out = symbol_ids.shape[1]

        encoder_mask = self.make_word_mask(lexical_token_ids)
        encoder_output = self.dropout(self.encode_words(lexical_token_ids, encoder_mask))

        decoder_mask = self.make_decoder_mask(b=b, n=s_out)
        atom_embeddings = self.atom_embedder(symbol_ids.to(self.device))

        extended_encoder_mask = encoder_mask.unsqueeze(1).repeat(1, s_out, 1).to(self.device)

        return encoder_output, atom_embeddings, decoder_mask, extended_encoder_mask

    def predict_atoms(self, reprs: Tensor, t: int) -> Tensor:
        return self.atom_embedder.invert(reprs, t)

    def encode_words(self, lexical_token_ids: LongTensor, encoder_mask: LongTensor) -> Tensor:
        encoder_output, _ = self.word_encoder(lexical_token_ids.to(self.device),
                                              attention_mask=encoder_mask.to(self.device))
        return encoder_output

    def encode_atoms(self, atom_reprs: Tensor, atom_mask: LongTensor, word_reprs: Tensor, word_mask: LongTensor) \
            -> Tensor:
        s_out = atom_reprs.shape[1]
        if s_out == 0:
            return atom_reprs
        return self.linker((self.dropout(word_reprs), word_mask, atom_reprs, atom_mask))[2]

    def link(self, atom_reprs: Tensor, atom_mask: LongTensor, word_reprs: Tensor, word_mask: LongTensor,
             pos_idxes: List[List[LongTensor]], neg_idxes: List[List[LongTensor]], exclude_singular: bool = True,
             sinkhorn_iters: int = 3) \
            -> List[Tensor]:

        atom_reprs = self.encode_atoms(atom_reprs, atom_mask, word_reprs, word_mask)

        _positives: List[List[Tensor]] = make_sinkhorn_inputs(atom_reprs, pos_idxes, self.device)
        _negatives: List[List[Tensor]] = make_sinkhorn_inputs(atom_reprs, neg_idxes, self.device)

        positives = [tensor for tensor in chain.from_iterable(_positives) if min(tensor.size()) != 0]
        negatives = [tensor for tensor in chain.from_iterable(_negatives) if min(tensor.size()) != 0]

        distinct_shapes = set(map(lambda tensor: tensor.size()[0], positives))
        if exclude_singular:
            distinct_shapes = distinct_shapes.difference({1})
        distinct_shapes = sorted(distinct_shapes)

        matches: List[Tensor] = []

        all_shape_positives: List[Tensor] \
            = [self.dropout(torch.stack([tensor for tensor in positives if tensor.size()[0] == shape]))
               for shape in distinct_shapes]
        all_shape_negatives: List[Tensor] \
            = [self.dropout(torch.stack([tensor for tensor in negatives if tensor.size()[0] == shape]))
               for shape in distinct_shapes]

        for this_shape_positives, this_shape_negatives in zip(all_shape_positives, all_shape_negatives):
            this_shape_positives = self.pos_transformation(this_shape_positives)
            this_shape_negatives = self.neg_transformation(this_shape_negatives)
            weights = torch.bmm(this_shape_positives,
                                this_shape_negatives.transpose(2, 1))
            matches.append(self.sinkhorn(weights, iters=sinkhorn_iters))
        return matches

    def link_train(self, *args, **kwargs) -> List[Tensor]:
        return self.link(*args, **kwargs, sinkhorn_iters=5)

    def link_eval(self, *args, **kwargs) -> List[Tensor]:
        return self.link(*args, **kwargs, sinkhorn_iters=5)

    def link_slow(self, atom_reprs: Tensor, atom_mask: LongTensor, word_reprs: Tensor, word_mask: LongTensor,
                  pos_idxes: List[List[LongTensor]], neg_idxes: List[List[LongTensor]], sinkhorn_iters: int = 10)\
            -> List[List[Tensor]]:

        atom_reprs = self.encode_atoms(atom_reprs, atom_mask, word_reprs, word_mask)

        _positives: List[List[Tensor]] = make_sinkhorn_inputs(atom_reprs, pos_idxes, self.device)
        _negatives: List[List[Tensor]] = make_sinkhorn_inputs(atom_reprs, neg_idxes, self.device)

        ret = []
        for sent in zip(_positives, _negatives):
            sent = list(zip(*sent))
            local = []
            for pos, neg in sent:
                pos = self.pos_transformation(self.dropout(pos))
                neg = self.neg_transformation(self.dropout(neg))
                weights = pos @ neg.transpose(-1, -2)
                local.append(self.sinkhorn(weights.unsqueeze(0), iters=sinkhorn_iters))
            ret.append(local)
        return ret

    def decode_train(self, lexical_token_ids: LongTensor, symbol_ids: LongTensor) \
            -> Tuple[Tensor, Tensor, Tensor, LongTensor]:

        tmp = self.precode(lexical_token_ids, symbol_ids)
        encoder_output, atom_embeddings, decoder_mask, extended_encoder_mask = tmp

        output_reprs = self.supertagger((encoder_output, extended_encoder_mask, atom_embeddings, decoder_mask))[2]

        return output_reprs, encoder_output, atom_embeddings, extended_encoder_mask

    def decode_greedy(self, lexical_token_ids: LongTensor, max_decode_length: Optional[int] = None,
                      length_factor: int = 5) -> Tuple[LongTensor, Tensor, Tensor, LongTensor, Tensor]:
        b, s_in = lexical_token_ids.shape

        if max_decode_length is None:
            s_out = length_factor * s_in
        else:
            s_out = max_decode_length

        encoder_mask = self.make_word_mask(lexical_token_ids)
        encoder_output = self.encode_words(lexical_token_ids, encoder_mask)

        extended_encoder_mask = self.extend_mask(encoder_mask, s_out).to(self.device)
        decoder_mask = self.make_decoder_mask(b, s_out)

        output_symbols = (torch.ones(b) * self.atom_tokenizer.sos_token_id).long().to(self.device)
        decoder_input = (self.atom_embedder.embed(output_symbols, 0)).unsqueeze(1)
        output_symbols = output_symbols.unsqueeze(1)
        decoder_output = torch.empty(b, s_out, self.dec_dim).to(self.device)

        for t in range(s_out):
            _decoder_tuple_input = (encoder_output, extended_encoder_mask,
                                    decoder_input, decoder_mask[:, :t + 1, :t + 1])
            repr_t = self.supertagger(_decoder_tuple_input)[2][:, -1]
            prob_t = self.predict_atoms(repr_t, t+1)
            class_t = prob_t.argmax(dim=-1)
            output_symbols = torch.cat([output_symbols, class_t.unsqueeze(1)], dim=1)
            next_embedding = self.atom_embedder.embed(class_t, t+1).unsqueeze(1)
            decoder_input = torch.cat([decoder_input, next_embedding], dim=1)
            decoder_output[:, t] = repr_t
        return output_symbols, decoder_input, encoder_output, extended_encoder_mask, decoder_output

    @torch.no_grad()
    def decode_beam(self, lexical_token_ids: LongTensor, beam_width: int, stop_at: ints,
                    max_decode_length: Optional[int] = None, length_factor: int = 5) \
            -> Tuple[LongTensor, Tensor, Tensor, LongTensor, Tensor]:

        def count_sep(x: Tensor, dim: int) -> Tensor:
            sep = self.atom_tokenizer.sep_token_id
            y = x == sep
            return y.sum(dim)

        def backward_index(idx: int) -> Tuple[int, int]:
            return idx // beam_width, idx - (idx // beam_width) * beam_width

        b, s_in = lexical_token_ids.shape

        stop_at = torch.tensor(stop_at, dtype=torch.long, device=self.device)

        if max_decode_length is None:
            s_out = length_factor * s_in
        else:
            s_out = max_decode_length

        encoder_mask = self.make_word_mask(lexical_token_ids)
        encoder_output = self.encode_words(lexical_token_ids, encoder_mask)

        extended_encoder_mask = encoder_mask.unsqueeze(1).repeat(1, s_out, 1).to(self.device)
        decoder_mask = self.make_decoder_mask(b, s_out)

        sos_tokens = (torch.ones(b) * self.atom_tokenizer.sos_token_id).long().to(self.device)
        decoder_input = (self.atom_embedder(sos_tokens)).unsqueeze(1)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, beam_width, 1, 1)
        sos_tokens = sos_tokens.unsqueeze(1)
        decoder_output = torch.zeros(b, beam_width, s_out, self.dec_dim).to(self.device)
        output_symbols = torch.zeros(b, beam_width, s_out, dtype=torch.long).to(self.device)
        beam_scores = torch.zeros(b, beam_width, device=self.device)

        _masked_probs = torch.ones(1, 1, len(self.atom_tokenizer), device=self.device) * -1e03
        _masked_probs[:, :, self.atom_tokenizer.pad_token_id] = 0

        for t in range(s_out):
            # the next repr of each batch-beam combination
            repr_t = torch.empty(b, beam_width, self.dec_dim, device=self.device)
            for beam in range(beam_width):
                _decoder_tuple_input = (encoder_output, extended_encoder_mask,
                                        decoder_input[:, beam], decoder_mask[:, :t + 1, :t + 1])
                repr_t[:, beam] = self.supertagger(_decoder_tuple_input)[2][:, -1]

            logprobs_t = self.predict_atoms(repr_t, t+1).log_softmax(dim=-1)

            sep_counts = count_sep(output_symbols, dim=-1)
            valid_beams = sep_counts < stop_at.unsqueeze(1)

            if not valid_beams.flatten().any():
                break

            if t == 0:
                logprobs_t[:, 1:] = -1e3  # hack first decoder step

            logprobs_t = torch.where(valid_beams.unsqueeze(-1), logprobs_t, _masked_probs)

            # lists of K tensors of shape B, K
            # local_scores[i] contains the K2 best branches originating from branch K1
            local_scores, local_steps = list(zip(*[logprobs_t[:, k].topk(k=beam_width, dim=-1)
                                                   for k in range(beam_width)]))

            # local_scores[b, k1, k2]  contains the total score of branch k2 originating from branch k1 in sent b
            local_scores = torch.stack(local_scores, dim=1) + beam_scores.unsqueeze(-1)
            local_steps = torch.stack(local_steps, dim=1)  # B, K1, K2
            best_scores, best_sources = local_scores.view(b, -1).topk(dim=-1, k=beam_width)  # B, K

            best_source_idxes: List[List[Tuple[int, int]]]  # B outer elements, K inner elements indexing (src, tgt)
            best_source_idxes = [[backward_index(idx) for idx in best_source] for best_source in best_sources.tolist()]

            new_decoder_output = torch.zeros(b, beam_width, s_out, self.dec_dim, device=self.device)
            new_decoder_input = torch.zeros(b, beam_width, t+1, self.dec_dim, device=self.device)
            new_output_symbols = torch.zeros(b, beam_width, s_out, dtype=torch.long, device=self.device)

            for sent in range(b):
                for beam in range(beam_width):
                    origin, target = best_source_idxes[sent][beam]

                    new_decoder_output[sent, beam, :t] = decoder_output[sent, origin, :t]
                    new_decoder_output[sent, beam, t] = repr_t[sent, origin]
                    new_decoder_input[sent, beam, :t+1] = decoder_input[sent, origin, :t+1]
                    new_output_symbols[sent, beam, :t] = output_symbols[sent, origin, :t]
                    new_output_symbols[sent, beam, t] = local_steps[sent, origin, target]
                    beam_scores[sent, beam] = best_scores[sent, beam]

            decoder_output = new_decoder_output
            decoder_input = new_decoder_input
            output_symbols = new_output_symbols

            class_t = output_symbols[:, :, t].view(b*beam_width)

            next_embedding = self.atom_embedder.embed(class_t, t+1).view(b, beam_width, self.dec_dim)

            if t != s_out - 1:
                next_embedding = next_embedding.unsqueeze(2)
                decoder_input = torch.cat([decoder_input, next_embedding], dim=2)
        output_symbols = torch.cat([sos_tokens.unsqueeze(1).repeat(1, beam_width, 1), output_symbols], dim=2)

        return (output_symbols, decoder_input, encoder_output, extended_encoder_mask,
                decoder_output.view(b, beam_width, -1, self.dec_dim))

    def train_batch(self, batch: Batch, loss_fn: Module, optimizer: Optimizer,
                    linking_weight: float = 0.5) -> Tuple[float, float]:
        self.train()

        words, types, pos_idxes, neg_idxes, grouped_permutors = batch
        atom_mask = self.make_atom_mask(types)

        output_reprs, encoder_output, atom_embeddings, word_mask = self.decode_train(words, types)

        # supertagging
        type_predictions = self.predict_atoms(output_reprs[:, :-1], 1)  # no predict on last token
        supertagging_loss = loss_fn(type_predictions.contiguous(), types[:, 1:].contiguous().to(self.device))

        if linking_weight == 0:
            supertagging_loss.backward()
            link_loss = 0
        else:
            # axiom linking
            link_weights = self.link_train(atom_embeddings, atom_mask, encoder_output, word_mask, pos_idxes, neg_idxes)
            grouped_permutors = [perm.to(self.device) for perm in grouped_permutors]
            link_loss = sum((
                functional.nll_loss(link.flatten(0, 1), perm.flatten(), reduction='mean')
                for link, perm in zip(link_weights, grouped_permutors)
            ))
            mutual_loss = link_loss * linking_weight + supertagging_loss
            mutual_loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        return supertagging_loss.item(), (link_loss.item() if link_loss != 0 else 0)

    @torch.no_grad()
    def eval_batch(self, batch: Batch, link: bool = True) \
            -> Tuple[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]:
        self.eval()

        words, types, pos_idxes, neg_idxes, grouped_permutors = batch
        atom_mask = self.make_atom_mask(types)
        true_embeddings = self.atom_embedder(types.to(self.device))

        max_length = types.shape[1]
        temp = self.decode_greedy(words, max_decode_length=max_length)
        type_predictions, _, encoder_output, wmask, _ = temp
        type_predictions = type_predictions[:, 1:-1]

        # supertagging
        types = types[:, 1:].to(self.device)

        if link:
            # linking
            links = self.link_eval(true_embeddings, atom_mask, encoder_output, wmask, pos_idxes, neg_idxes, False)
            permutors = [perm.to(self.device) for perm in grouped_permutors]

            return (measure_linking_accuracy(links, permutors),
                    measure_supertagging_accuracy(type_predictions, types))
        return (0, 1), measure_supertagging_accuracy(type_predictions, types)

    def train_epoch(self, dataloader: DataLoader, loss_fn: Module, optimizer: Optimizer,
                    linking_weight: float = 0.5) -> Tuple[float, float]:

        total_l1, total_l2 = 0, 0
        for batch in tqdm(dataloader):
            l1, l2 = self.train_batch(batch, loss_fn, optimizer, linking_weight=linking_weight)
            total_l1 += l1
            total_l2 += l2
        return total_l1 / len(dataloader), total_l2 / len(dataloader)

    def eval_epoch(self, dataloader: DataLoader, link: bool = True) -> Tuple[float, float, float]:
        l_total, l_correct, s_total, s_correct, w_total, w_correct = (0.1,) * 6

        for batch in tqdm(dataloader):
            batch_output = self.eval_batch(batch, link)
            (bl_correct, bl_total), ((bs_correct, bs_total), (bw_correct, bw_total)) = batch_output
            s_total += bs_total
            s_correct += bs_correct
            w_total += bw_total
            w_correct += bw_correct
            l_total += bl_total
            l_correct += bl_correct
        return s_correct / s_total, w_correct / w_total, l_correct / l_total

    @torch.no_grad()
    def infer(self, sents: strs, beam_size: int, **kwargs) -> List[List[Analysis]]:
        self.eval()

        sent_lens = [len(sent.split()) + 1 for sent in sents]
        lexical_token_ids = sents_to_batch(sents, self.tokenizer)

        if beam_size == 1:
            temp = self.decode_greedy(lexical_token_ids.to(self.device), **kwargs)
            type_preds, atom_embeddings, encoder_output, wmask_, _ = temp
            type_preds = [[x] for x in type_preds.tolist()]
            atom_embeddings = atom_embeddings.unsqueeze(1)
        else:
            temp = self.decode_beam(lexical_token_ids.to(self.device), beam_width=beam_size,
                                    stop_at=sent_lens, **kwargs)
            type_preds, atom_embeddings, encoder_output, wmask_, _ = temp
            type_preds = type_preds.tolist()

        atom_seqs: List[List[Optional[List[strs]]]]
        # filter decoded atoms that count at least as many types as words
        atom_seqs = self.atom_tokenizer.convert_beam_ids_to_polish(type_preds, sent_lens)

        analyses = self.type_parser.analyze_beam_batch(sents, atom_seqs)
        valid_for_linking = [((s, b), len(analyses[s][b].polish), analyses[s][b])
                             for s in range(len(analyses))
                             for b in range(len(analyses[s]))
                             if analyses[s][b].polish is not None]
        if not valid_for_linking:
            return analyses

        ids, atom_lens, valid_analyses = list(zip(*valid_for_linking))
        atom_mask = self.make_atom_mask_from_lens(atom_lens)

        atom_reprs = torch.zeros(len(ids), max(atom_lens), self.dec_dim, device=self.device)
        word_reprs = torch.zeros(len(ids), encoder_output.shape[1], self.enc_dim, device=self.device)
        wmask = torch.zeros(len(ids), wmask_.shape[1], wmask_.shape[2], device=self.device, dtype=torch.long)
        for i, (s, b) in enumerate(ids):
            atom_reprs[i] = atom_embeddings[s, b, :max(atom_lens)]
            word_reprs[i] = encoder_output[s]
            wmask[i] = wmask_[s]

        positive_ids, negative_ids = self.type_parser.analyses_to_indices(valid_analyses)

        links_ = self.link_slow(atom_reprs, atom_mask, word_reprs, wmask, positive_ids, negative_ids)

        links = [[link_weights.argmax(dim=-1).tolist()[0] for link_weights in sent] for sent in links_]
        for va, link in zip(valid_analyses, links):
            va.fill_matches(link)

        return analyses

    @torch.no_grad()
    def parse_with_oracle(self, samples: List[Sample]) -> List[List[Analysis]]:
        self.eval()

        sent_lens = list(map(lambda s: len(s.words) + 1, samples))
        words, types, pos_ids, neg_ids = self.atom_tokenizer.samples_to_batch(samples, self.tokenizer)

        tmp = self.precode(words, types)
        encoder_output, atom_embeddings, decoder_mask, wmask_ = tmp
        type_preds = [[x] for x in types.tolist()]
        atom_embeddings = atom_embeddings.unsqueeze(1)
        atom_seqs: List[List[Optional[List[strs]]]]
        # filter decoded atoms that count at least as many types as words
        atom_seqs = self.atom_tokenizer.convert_beam_ids_to_polish(type_preds, sent_lens)

        analyses = self.type_parser.analyze_beam_batch([' '.join(s.words) for s in samples], atom_seqs)
        valid_for_linking = [((s, b), len(analyses[s][b].polish), analyses[s][b])
                             for s in range(len(analyses))
                             for b in range(len(analyses[s]))
                             if analyses[s][b].polish is not None]
        if not valid_for_linking:
            return analyses

        ids, atom_lens, valid_analyses = list(zip(*valid_for_linking))
        atom_mask = self.make_atom_mask_from_lens(atom_lens)

        atom_reprs = torch.zeros(len(ids), max(atom_lens), self.dec_dim, device=self.device)
        word_reprs = torch.zeros(len(ids), encoder_output.shape[1], self.enc_dim, device=self.device)
        wmask = torch.zeros(len(ids), wmask_.shape[1], wmask_.shape[2], device=self.device, dtype=torch.long)
        for i, (s, b) in enumerate(ids):
            atom_reprs[i] = atom_embeddings[s, b, :max(atom_lens)]
            word_reprs[i] = encoder_output[s]
            wmask[i] = wmask_[s]

        positive_ids, negative_ids = self.type_parser.analyses_to_indices(valid_analyses)

        links_ = self.link_slow(atom_reprs, atom_mask, word_reprs, wmask, positive_ids, negative_ids)

        links = [[link_weights.argmax(dim=-1).tolist()[0] for link_weights in sent] for sent in links_]
        for va, link in zip(valid_analyses, links):
            va.fill_matches(link)
        return analyses
