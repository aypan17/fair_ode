# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from collections import defaultdict

import time
import abc
import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch_scatter

from src.envs.char_sp import BinaryEqnTree, SYMBOL_ENCODER, EOS, LEAF, VOCAB, BINARY, UNARY, \
                            DERIVATIVES, DIFFERENTIALS, INT, DIGITS

from src.model.modules import *

N_MAX_POSITIONS = 4096  # maximum input sequence length


logger = getLogger()


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


class MultiHeadAttention(nn.Module):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, dropout):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)

    def forward(self, input, mask, kv=None, cache=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.size(1)
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(input))                                          # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))                                      # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))                                      # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))                                          # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))                                          # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)                             # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)                             # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(dim_per_head)                                       # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))                           # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)               # (bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, -float('inf'))                              # (bs, n_heads, qlen, klen)

        weights = F.softmax(scores.float(), dim=-1).type_as(scores)           # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)
        context = torch.matmul(weights, v)                                    # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)                                            # (bs, qlen, dim)

        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = weights.detach().cpu()

        return self.out_lin(context)


class TransformerFFN(nn.Module):

    def __init__(self, in_dim, dim_hidden, out_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)

    def forward(self, input):
        x = self.lin1(input)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerModel(nn.Module):

    STORE_OUTPUTS = False

    def __init__(self, params, id2word, is_encoder, with_output):
        """
        Transformer model (encoder or decoder).
        """
        super().__init__()

        # encoder / decoder, output layer
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output

        # dictionary
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.id2word = id2word
        assert len(self.id2word) == self.n_words

        # model parameters
        self.dim = params.emb_dim       # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = params.n_heads   # 8 by default
        self.n_layers = params.n_enc_layers if is_encoder else params.n_dec_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'

        # embeddings
        self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        if params.sinusoidal_embeddings:
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight)
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-12)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-12))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-12))
                self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, dropout=self.dropout))
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-12))

        # output layer
        if self.with_output:
            self.proj = nn.Linear(self.dim, params.n_words, bias=True)
            if params.share_inout_emb:
                self.proj.weight = self.embeddings.weight

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(self, x, lengths, causal, src_enc=None, src_len=None, positions=None, cache=None, previous_state=None):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
        """
        # lengths = (x != self.pad_index).float().sum(dim=1)
        # mask = x != self.pad_index

        # check inputs
        slen, bs = x.size()
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        x = x.transpose(0, 1)  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.size(0) == bs
        assert previous_state is None or cache is None

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)
        if self.is_decoder and src_enc is not None:
            src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]

        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # do not recompute cached elements
        if cache is not None:
            _slen = slen - cache['slen']
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # all layer outputs
        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs = []

        # embeddings
        if previous_state is None:
            tensor = self.embeddings(x)
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
            tensor = self.layer_norm_emb(tensor)
            tensor = F.dropout(tensor, p=self.dropout, training=self.training)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        else:
            assert previous_state.shape == (slen, bs, self.dim)
            tensor = previous_state.transpose(0, 1)
        if TransformerModel.STORE_OUTPUTS and not self.training:
            self.outputs.append(tensor.detach().cpu())

        # transformer layers
        for i in range(self.n_layers):

            # self attention
            attn = self.attentions[i](tensor, attn_mask, cache=cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            if self.is_decoder and src_enc is not None:
                attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
            if TransformerModel.STORE_OUTPUTS and not self.training:
                self.outputs.append(tensor.detach().cpu())

        # update cache length
        if cache is not None:
            cache['slen'] += tensor.size(1)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor

    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        assert (y == self.pad_index).sum().item() == 0
        scores = self.proj(x).view(-1, self.n_words)
        loss = F.cross_entropy(scores, y, reduction='mean')
        return scores, loss

    def generate(self, src_enc, src_len, max_len=200, sample_temperature=None):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        """

        # input batch
        bs = len(src_len)
        assert src_enc.size(0) == bs

        # generated sentences
        generated = src_len.new(max_len, bs)  # upcoming output
        generated.fill_(self.pad_index)       # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)    # we use <EOS> for <BOS> everywhere

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        gen_len = src_len.clone().fill_(1)
        unfinished_sents = src_len.clone().fill_(1)

        # cache compute states
        cache = {'slen': 0}

        while cur_len < max_len:

            # compute word scores
            tensor = self.forward(
                'fwd',
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                cache=cache
            )
            assert tensor.size() == (1, bs, self.dim), (cur_len, max_len, src_enc.size(), tensor.size(), (1, bs, self.dim))
            tensor = tensor.data[-1, :, :].type_as(src_enc)  # (bs, dim)
            scores = self.proj(tensor)                       # (bs, n_words)

            # select next words: sample or greedy
            if sample_temperature is None:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            else:
                next_words = torch.multinomial(F.softmax(scores / sample_temperature, dim=1), 1).squeeze(1)
            assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            generated[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            gen_len.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)

        # sanity check
        assert (generated == self.eos_index).sum() == 2 * bs

        return generated[:cur_len], gen_len

    def generate_beam(self, src_enc, src_len, beam_size, length_penalty, early_stopping, max_len=200):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        """

        # check inputs
        assert src_enc.size(0) == src_len.size(0)
        assert beam_size >= 1

        # batch size / number of words
        bs = len(src_len)
        n_words = self.n_words

        # expand to beam size the source latent representations / source lengths
        src_enc = src_enc.unsqueeze(1).expand((bs, beam_size) + src_enc.shape[1:]).contiguous().view((bs * beam_size,) + src_enc.shape[1:])
        src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(max_len, bs * beam_size)  # upcoming output
        generated.fill_(self.pad_index)                   # fill upcoming ouput with <PAD>
        generated[0].fill_(self.eos_index)                # we use <EOS> for <BOS> everywhere

        # generated hypotheses
        generated_hyps = [BeamHypotheses(beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]

        # positions
        positions = src_len.new(max_len).long()
        positions = torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # current position
        cur_len = 1

        # cache compute states
        cache = {'slen': 0}

        # done sentences
        done = [False for _ in range(bs)]

        while cur_len < max_len:

            # compute word scores
            tensor = self.forward(
                'fwd',
                x=generated[:cur_len],
                lengths=src_len.new(bs * beam_size).fill_(cur_len),
                positions=positions[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                cache=cache
            )
            assert tensor.size() == (1, bs * beam_size, self.dim)
            tensor = tensor.data[-1, :, :]          # (bs * beam_size, dim)
            scores = self.proj(tensor)              # (bs * beam_size, n_words)
            scores = F.log_softmax(scores, dim=-1)  # (bs * beam_size, n_words)
            assert scores.size() == (bs * beam_size, n_words)

            # select next words with scores
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (bs * beam_size, n_words)
            _scores = _scores.view(bs, beam_size * n_words)            # (bs, beam_size * n_words)

            next_scores, next_words = torch.topk(_scores, 2 * beam_size, dim=1, largest=True, sorted=True)
            assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            # for each sentence
            for sent_id in range(bs):

                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item())
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_index, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # end of sentence, or next word
                    if word_id == self.eos_index or cur_len + 1 == max_len:
                        generated_hyps[sent_id].add(generated[:cur_len, sent_id * beam_size + beam_id].clone().cpu(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert len(next_sent_beam) == 0 if cur_len + 1 == max_len else beam_size
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_index, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * (sent_id + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == bs * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])

            # re-order batch and internal states
            generated = generated[:, beam_idx]
            generated[cur_len] = beam_words
            for k in cache.keys():
                if k != 'slen':
                    cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # def get_coeffs(s):
        #     roots = [int(s[i + 2]) for i, c in enumerate(s) if c == 'x']
        #     poly = np.poly1d(roots, r=True)
        #     coeffs = list(poly.coefficients.astype(np.int64))
        #     return [c % 10 for c in coeffs], coeffs

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         hh = " ".join(self.id2word[x] for x in ww.tolist())
        #         print(f"{ss:+.4f} {hh}")
        #         # cc = get_coeffs(hh[4:])
        #         # print(f"{ss:+.4f} {hh} || {cc[0]} || {cc[1]}")
        #     print("")

        # select the best hypotheses
        tgt_len = src_len.new(bs)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = src_len.new(tgt_len.max().item(), bs).fill_(self.pad_index)
        for i, hypo in enumerate(best):
            decoded[:tgt_len[i] - 1, i] = hypo
            decoded[tgt_len[i] - 1, i] = self.eos_index

        # sanity check
        assert (decoded == self.eos_index).sum() == 2 * bs

        return decoded, tgt_len, generated_hyps


class BeamHypotheses(object):

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty

'''
class UnaryLSTM(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.data = nn.Linear(d_model, d_model, bias=True)
        self.forget = nn.Linear(d_model, d_model, bias=True)
        self.output = nn.Linear(d_model, d_model, bias=True)
        self.input = nn.Linear(d_model, d_model, bias=True)

    def forward(self, inp: torch.Tensor, dropout=None) -> torch.Tensor:
        h, c = torch.split(inp, 1, dim=1)
        h = h.squeeze(1)
        c = c.squeeze(1)
        i = torch.sigmoid(self.data(h))
        f = torch.sigmoid(self.forget(h))
        o = torch.sigmoid(self.output(h))
        u = torch.tanh(self.input(h))
        if dropout is None:
            c = i * u + f * c
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f * c
        h = o * torch.tanh(c)
        h_norm = h / h.norm(p=2)
        c_norm = c / c.norm(p=2)
        return torch.stack([h_norm, c_norm], dim=1)


class BinaryLSTM(torch.nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.data_left = nn.Linear(d_model, d_model, bias=False)
        self.data_right = nn.Linear(d_model, d_model, bias=False)
        self.data_bias = nn.Parameter(torch.FloatTensor([0] * d_model))
        self.forget_left_by_left = nn.Linear(d_model, d_model, bias=False)
        self.forget_left_by_right = nn.Linear(d_model, d_model, bias=False)
        self.forget_right_by_left = nn.Linear(d_model, d_model, bias=False)
        self.forget_right_by_right = nn.Linear(d_model, d_model, bias=False)
        self.forget_bias_left = nn.Parameter(torch.FloatTensor([0] * d_model))
        self.forget_bias_right = nn.Parameter(torch.FloatTensor([0] * d_model))
        self.output_left = nn.Linear(d_model, d_model, bias=False)
        self.output_right = nn.Linear(d_model, d_model, bias=False)
        self.output_bias = nn.Parameter(torch.FloatTensor([0] * d_model))
        self.input_left = nn.Linear(d_model, d_model, bias=False)
        self.input_right = nn.Linear(d_model, d_model, bias=False)
        self.input_bias = nn.Parameter(torch.FloatTensor([0] * d_model))

    def forward(self, inp_left, inp_right, dropout=None):
        """

        Args:
            input_left: ((num_hidden,), (num_hidden,))
            input_right: ((num_hidden,), (num_hidden,))

        Returns:
            (num_hidden,), (num_hidden)
        """
        hl, cl = torch.split(inp_left, 1, dim=1)
        hr, cr = torch.split(inp_right, 1, dim=1)
        hl = hl.squeeze(1)
        cl = cl.squeeze(1)
        hr = hr.squeeze(1)
        cr = cr.squeeze(1)
        i = torch.sigmoid(self.data_left(hl) + self.data_right(hr) + self.data_bias)
        f_left = torch.sigmoid(self.forget_left_by_left(hl) + self.forget_left_by_right(hr) + self.forget_bias_left)
        f_right = torch.sigmoid(self.forget_right_by_left(hl) + self.forget_right_by_right(hr) + self.forget_bias_right)
        o = torch.sigmoid(self.output_left(hl) + self.output_right(hr) + self.output_bias)
        u = torch.tanh(self.input_left(hl) + self.input_right(hr) + self.input_bias)
        if dropout is None:
            c = i * u + f_left * cl + f_right * cr
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f_left * cl + f_right * cr
        h = o * torch.tanh(c)
        h_norm = h / h.norm(p=2)
        c_norm = c / c.norm(p=2)
        return torch.stack([h_norm, c_norm], dim=1)

'''
class TreeNN(torch.nn.Module):
    def __init__(self, params, id2word, word2id, una_ops, bin_ops):
        super().__init__()
        self.d_model = params.emb_dim
        self.id2word = id2word
        self.word2id = word2id
        self.una_ops = una_ops
        self.bin_ops = bin_ops
        self.pad_idx = params.pad_index
        self.leaf_emb = torch.nn.Embedding(
            num_embeddings=len(id2word),
            embedding_dim=params.emb_dim,
            padding_idx=self.pad_idx,
            max_norm=1.0,
        )
        self.num_enc = torch.nn.Sequential(
            nn.Linear(1, params.emb_dim),
            nn.Sigmoid(),
            nn.Linear(params.emb_dim, params.emb_dim),
            nn.Sigmoid()
        )

    @abc.abstractmethod
    def _apply_function(self, function_name: str, input_cell: torch.Tensor, input_hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute the activation for an internal node of the tree.

        Parameters
        ----------
        function_name
            Name of the function to apply.
        inputs
            Inputs to each module in the current step.
            Has shape [batch_size, function_arity, self.hparams.d_model],
            where `batch_size` is the number of nodes being computed at this step.

        Returns
        -------
        torch.Tensor
            The vector to pass up the tree.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_output(self, inputs, lens):
        """
        Compute the outputs for this model.

        Parameters
        ----------
        inputs
            Inputs to each root module in the batch.
            Has shape [batch_size, root_function_arity, self.hparams.d_model].
        lens
            Number of embeddings/nodes of each graph in batch.
            Has shape [batch_size].
        pad
            Whether to pad the output with start and end tokens.

        Returns
        -------
        torch.Tensor
            Whatever output this model produces.
        """
        batch = torch.split(inputs, lens.tolist(), dim=0)
        return pad_sequence(batch, padding_value=0.0, batch_first=True).squeeze(2)

    def forward(self, x, lengths, seq_num=False):
        """
        Given a (possibly batched) graph and the number of nodes per tree, 
        compute the outputs for each tree.
        """
        operations, tokens, left_idx, right_idx, depths, operation_order, _, integers, int_lens = x
        num_steps = operation_order.numel()
        num_nodes = operations.numel()

        # A buffer where the i-th row is the activations output from the i-th node
        # The buffer is repeatedly summed into to allow dense gradient computation;
        # this is valid because each position is summed to exactly once.
        activations = torch.zeros(
            (num_nodes, self.d_model), device=tokens.device
        )

        # A buffer where the i-th row is the memory output from the i-th node.
        memory = torch.zeros(
            (num_nodes, self.memory_size, self.d_model), device=tokens.device
        )

        for depth in range(num_steps):  
            step_mask = depths == depth  # Indices to compute at this step
            op = operation_order[depth].item()

            if op == -1: # Embedding lookup or number encoding
                leaf_tokens = tokens.masked_select(step_mask)
                step_activations = self.leaf_emb(leaf_tokens)
                activations = activations.masked_scatter(
                    step_mask.unsqueeze(1), step_activations
                )
            else:
                op_name = self.id2word[op]
                step_ids = torch.nonzero(step_mask, as_tuple=True)[0]
                left = left_idx.masked_select(step_mask)
                right = right_idx.masked_select(step_mask) # if we have a unary function this is meaningless
                predecessors = torch.stack((left, right), dim=1)
                input_cell = activations[predecessors]
                input_hidden = memory[predecessors]

                step_activations, step_memory = self._apply_function(
                    op_name, input_cell, input_hidden
                )
                activation_scatter = torch_scatter.scatter(
                    src=step_activations, index=step_ids, dim=0, dim_size=num_nodes
                )
                memory_scatter = torch_scatter.scatter(
                    src=step_memory, index=step_ids, dim=0, dim_size=num_nodes
                )
                activations = activations + activation_scatter
                memory = memory + memory_scatter

        # Reverse activations because nodes are listed in reverse pre-order.
        return self._compute_output(activations, lengths)

class TreeLSTM_Encoder(TreeNN):
    """
    A TreeLSTM model.

    For full parameters, see the docstring for `RecursiveNN`.
    """

    def __init__(self, params, id2word, word2id, una_ops, bin_ops):
        super().__init__(params, id2word, word2id, una_ops, bin_ops)

        self.unary_function_modules = torch.nn.ModuleDict(
            {f: UnaryLSTM(self.d_model) for f in una_ops}
        )
        self.binary_function_modules = torch.nn.ModuleDict(
            {f: BinaryLSTM(self.d_model) for f in bin_ops}
        )
        self.memory_size = 1

    def _apply_function(self, function_name, inputs, memory):
        if function_name in self.una_ops:
            module = self.unary_function_modules[function_name]
            inputs = inputs[:, 0, :]
            memory = memory[:, 0, :]
            return module(inputs, memory)

        if function_name in self.bin_ops:
            # Concatenate left and right before function application
            module = self.binary_function_modules[function_name]
            l_inputs = inputs[:, 0, :]
            r_inputs = inputs[:, 1, :]
            l_memory = memory[:, 0, :]
            r_memory = memory[:, 1, :]
            return module(l_inputs, r_inputs, l_memory, r_memory)

        assert False


class TreeSMU_Encoder(TreeNN):
    """
    A TreeSMU model.

    For full parameters, see the docstring for `RecursiveNN`.
    # TODO: implement
    """

    def __init__(self, params, id2word, word2id, una_ops, bin_ops):
        super().__init__(params, id2word, word2id, una_ops, bin_ops)

        self.unary_function_modules = torch.nn.ModuleDict(
            {f: UnarySMU(params) for f in una_ops}
        )
        self.unary_stack_modules = torch.nn.ModuleDict(
            {f: UnaryStack(params) for f in una_ops}
        )
        self.binary_function_modules = torch.nn.ModuleDict(
            {f: BinarySMU(params) for f in bin_ops}
        )
        self.binary_stack_modules = torch.nn.ModuleDict(
            {f: BinaryStack(params) for f in bin_ops}
        )
        self.memory_size = params.stack_size

    def _apply_function(self, function_name, inputs, memory):
        if function_name in self.una_ops:
            module = self.unary_function_modules[function_name]
            stack = self.unary_stack_modules[function_name]
            inputs = inputs[:, 0, :]
            memory = memory[:, 0, :]
            step_memory = stack(inputs, memory)
            return module(inputs, step_memory).squeeze(1), step_memory

        if function_name in self.bin_ops:
            # Concatenate left and right before function application
            module = self.binary_function_modules[function_name]
            stack = self.binary_stack_modules[function_name]
            l_inputs = inputs[:, 0, :]
            r_inputs = inputs[:, 1, :]
            l_memory = memory[:, 0, :]
            r_memory = memory[:, 1, :]
            step_memory = stack(l_inputs, r_inputs, l_memory, r_memory)
            return module(l_inputs, r_inputs, step_memory).squeeze(1), step_memory

        assert False

class TreeLSTM_Encoders(torch.nn.Module):
    def __init__(self, params, id2word, word2id, una_ops, bin_ops):
        super().__init__()

        self.d_model = params.emb_dim
        self.id2word = id2word
        self.word2id = word2id
        self.una_ops = una_ops
        self.bin_ops = bin_ops
        self.pad_idx = params.pad_index
        self.unary_modules = torch.nn.ModuleDict(
            {f: UnaryLSTM(params.emb_dim) for f in una_ops}
        )
        self.binary_modules = torch.nn.ModuleDict(
            {f: BinaryLSTM(params.emb_dim) for f in bin_ops}
        )
        self.leaf_emb = torch.nn.Embedding(
            num_embeddings=len(id2word),
            embedding_dim=params.emb_dim,
            padding_idx=self.pad_idx,
            max_norm=1.0,
        )
        self.num_enc = torch.nn.Sequential(
            nn.Linear(1, params.emb_dim),
            nn.Sigmoid(),
            nn.Linear(params.emb_dim, params.emb_dim),
            nn.Sigmoid()
        )
        if params.character_rnn:
            self.ch_rnn = torch.nn.RNN(params.emb_dim, params.emb_dim, dropout=params.dropout)

    '''
    operations: torch.Tensor,
    tokens: torch.Tensor,
    left_idx: torch.Tensor,
    right_idx: torch.Tensor,
    depths: torch.Tensor,
    operation_order: torch.Tensor,
    digits: torch.Tensor,
    integers: torch.Tensor
    '''
    def forward(self, x=None, lengths=None, augment=False, seq_num=False):
        if augment:
            return self.forward_augment(x)
        return self.forward_(x, lengths, seq_num)

    def forward_(
        self, x, lengths, seq_num
    ) -> torch.Tensor:
        """
        Given a batch of tensorized trees, encode the trees and return the hidden state of each node.
        """
        operations, tokens, left_idx, right_idx, depths, operation_order, _, integers, int_lens = x
        num_steps = operation_order.numel()
        num_nodes = operations.numel()
        activations = torch.zeros(
            (num_nodes, 2, self.d_model), dtype=torch.float, device=operations.device
        )

        if seq_num:
            int_emb = torch.zeros(
                (num_nodes, max(int_lens), self.d_model), dtype=torch.float, device=operations.device
            )
            int_mask = torch.zeros(num_nodes, dtype=torch.bool, device=operations.device)

        s = time.time()
        for depth in range(num_steps):  # type: ignore
            step_mask = depths == depth  # Indices to compute at this step
            op = operation_order[depth].item()

            if op in [-1, -2]: # Embedding lookup or number encoding
                idx = tokens.masked_select(step_mask)
                step_activations = self.leaf_emb(idx) if op == -1 else self.num_enc(idx.float().unsqueeze(1))
                zeros = torch.zeros(
                    (len(idx), self.d_model), dtype=torch.float, device=operations.device
                )
                step_activations = torch.stack([step_activations, zeros], dim=1)

                if op == -2 and seq_num:
                    int_mask = step_mask
                    int_emb = int_emb.masked_scatter(
                        int_mask.unsqueeze(1).unsqueeze(1), self.rnn(integers, int_lens)
                    )

            else:
                op_name = self.id2word[op]

                if op_name in self.unary_modules.keys():
                    module = self.unary_modules[op_name]
                    child_input_idx = left_idx.masked_select(step_mask)
                    child_activations = activations[child_input_idx]
                    step_activations = module(child_activations)
                else:  # Binary; equality operations always have a depth of -1
                    module = self.binary_modules[op_name]
                    left_input_idx = left_idx.masked_select(step_mask)
                    left_activations = activations[left_input_idx]
                    right_input_idx = right_idx.masked_select(step_mask)
                    right_activations = activations[right_input_idx]
                    step_activations = module(left_activations, right_activations)

            # Write computed activations into the shared buffer; NOTE: one copy of this
            # buffer is computed for each step, to allow for dense backprop
            activations = activations.masked_scatter(
                torch.stack([step_mask, step_mask], dim=1).unsqueeze(2), step_activations
            )
        e = time.time()
        print("Forward:", e-s)
        hidden, _ = torch.split(activations, 1, dim=1)

        if seq_num:
            s = time.time()
            zeros = torch.zeros(
                (num_nodes, max(int_lens)-1, self.d_model), dtype=torch.float, device=hidden.device
            )
            hidden_pad = torch.cat([hidden, zeros], dim=1)
            hidden_pad = torch.where(int_mask.unsqueeze(1).unsqueeze(1), int_emb, hidden_pad)
            dim = hidden_pad.size()
            hidden_pad_flat = hidden_pad.view(dim[0]*dim[1], self.d_model)
            nonzero = (hidden_pad_flat != 0).all(1)
            hidden = hidden_pad_flat[nonzero, :]
            e = time.time()
            print("Replace emb:", e-s)

        unpadded_batch = torch.split(hidden, lengths.tolist(), dim=0)
        return pad_sequence(unpadded_batch, padding_value=0.0, batch_first=True).squeeze(2)

    '''
    Computes a batch of equations with the form Y' - k*exp(x), Y=k*exp(x) for k in integers. 
    Augments the training for the NUMBER_ENCODER block.
    '''
    def forward_augment(self, x):
        # Intialize cell states
        zeros = torch.zeros((len(x), 1, self.d_model), dtype=torch.float, device=x.device)

        # Y' and x and EOS embeddings
        eos = self.word2id['<s>'] * torch.ones((len(x), 1), dtype=torch.long, device=x.device)
        y_token = self.word2id["Y'"] * torch.ones((len(x), 1), dtype=torch.long, device=x.device)
        x_token = self.word2id['x'] * torch.ones((len(x), 1), dtype=torch.long, device=x.device)
        eos, y_token, x_token = torch.split(self.leaf_emb(torch.cat([y_token, x_token, eos], dim=1)), 1, dim=1)

        # exp(x) embedding
        exp_module = self.unary_modules['exp']
        exp = exp_module(torch.cat([x_token, zeros], dim=1))

        # number embedding
        int_emb = self.num_enc(x.unsqueeze(1)).unsqueeze(1)

        # mul embedding
        mul_module = self.binary_modules['mul']
        mul = mul_module(torch.cat([int_emb, zeros], dim=1), exp)

        # sub embedding
        sub_module = self.binary_modules['sub']
        sub = sub_module(torch.cat([x_token, zeros], dim=1), mul)

        # split into hidden, cell
        exp, _ = torch.split(exp, 1, dim=1) 
        mul, _ = torch.split(mul, 1, dim=1)
        sub, _ = torch.split(sub, 1, dim=1)

        return torch.cat([eos, sub, y_token, mul, int_emb, exp, x_token, eos], dim=1)

    def rnn(self, integers, int_lens):
        s = time.time()
        emb = self.leaf_emb(integers)
        packed_inp = pack_padded_sequence(emb, int_lens.cpu().numpy(), batch_first=True, enforce_sorted=False)
        packed_out, cell = self.ch_rnn(packed_inp)
        output, _ = pad_packed_sequence(packed_out, batch_first=True)
        e = time.time()
        print("RNN emb:", e-s)
        return output
