# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from collections import defaultdict

import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from src.envs.char_sp import BinaryEqnTree, SYMBOL_ENCODER, EOS, LEAF, VOCAB, BINARY, UNARY, \
                            DERIVATIVES, DIFFERENTIALS, INT, DIGITS

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
class TreeLSTM_Encoder(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dim = params.emb_dim    
        self.dropout = params.dropout 
        self.cpu = params.cpu
        self.symmetric = params.symmetric # Whether or not to use symmetric blocks for add/mul
        self.max_order = params.order # Highest derivative that appears in the ode (currently 2)
        self.num_vars = params.vars # Number of variables in the ODE (currently 1)
        self.num_bit = params.num_bit # Max number of bits allowed in binary represenation

        # Symbol embeddings
        setattr(self, SYMBOL_ENCODER, nn.Embedding(num_embeddings=len(VOCAB), embedding_dim=self.dim))

        # Binary embeddings
        self.bin2emb = nn.Embedding(num_embeddings=self.num_bit, embedding_dim=self.dim)

        # Integer encoding block
        for sign in INT:
            setattr(self, sign, nn.Sequential(nn.Linear(self.dim, self.dim), nn.Tanh(), nn.Linear(self.dim, self.dim)))

        # EOS encoding block
        setattr(self, '#', UnaryLSTMNode(self.dim, self.dim))

        # Derivative operators
        for d in DIFFERENTIALS:
            setattr(self, d, UnaryLSTMNode(self.dim, self.dim))

        # Unary operators
        for op in UNARY:
            setattr(self, op, UnaryLSTMNode(self.dim, self.dim))

        # Binary operators
        for op in BINARY:
            if self.symmetric and (op == 'add' or op == 'mul'):
                setattr(self, op, BinaryLSTMNodeSym(num_input=self.dim, num_hidden=self.dim))
            else:
                setattr(self, op, BinaryLSTMNode(num_input=self.dim, num_hidden=self.dim))

        # Not sure how to add bias in a sensible manner
        # self.bias = nn.Parameter(torch.FloatTensor([0]))

        
    # We don't use attention yet, so casual is set to False
    def forward(self, x=None, causal=False):
        batch, id2emb, lengths, invalid = self.prefix_to_tree(x)
        children, parents, embeddings, buckets, max_depth, batch_size, roots = self.label_and_map_batch(batch)

        # Compute nodes
        for depth in range(1, max_depth + 1):
            input_ids = buckets[depth]
            inputs, ids = self.concatenate_inputs(self.dim, input_ids, parents, embeddings)

            for key in inputs:
                nn_block = getattr(self, key)
                if key == SYMBOL_ENCODER or key in INT:
                    hidden, cell = inputs[key]
                    output = (nn_block(hidden), cell)
                elif key == EOS or key in UNARY or key in DIFFERENTIALS:
                    hidden, cell = inputs[key]
                    output = nn_block((hidden, cell), dropout=self.dropout)
                elif key in BINARY:
                    output = nn_block(inputs[key][0], inputs[key][1], dropout=self.dropout)
                else:
                    raise AssertionError("The given key is not valid. Key:", key)

                # For a given key, set the embedding for each id in ids[key] in sequential order.
                for j in range(len(ids[key])):
                    embeddings[ids[key][j]] = (output[0][j], output[1][j])

        max_len = torch.max(lengths).item()
        output = self.flatten_tree(embeddings, id2emb, max_len)
        assert list(output.size()) == [batch_size, max_len, self.dim]
        return output, lengths, invalid

    # Helper function for concatenating inputs for models with a hidden state and cell state. 
    # For fixed depth, takes a dictionary of idx of child nodes and their embeddings.
    # Returns a concatenated tuple of (hidden state, cell state) and a dictionary of parent ids, indexed by key.
    def concatenate_inputs(self, dim, input_ids, parents, embeddings):
        inputs = {}
        ids = {}
        for key in input_ids:
            if key == SYMBOL_ENCODER or key in INT: 
                children = input_ids[key][0]
                hidden_state = torch.stack([embeddings[child_id] for child_id in children], dim=0)
                cell_state = torch.zeros((len(children), dim), dtype=torch.float)
                cell_state = cell_state if self.cpu else cell_state.cuda()
                inputs[key] = hidden_state, cell_state
            elif key == EOS or key in UNARY or key in DIFFERENTIALS:
                children = input_ids[key][0]
                hidden_state = torch.stack([embeddings[child_id][0] for child_id in children], dim=0)
                cell_state = torch.stack([embeddings[child_id][1] for child_id in children], dim=0)
                inputs[key] = hidden_state, cell_state
            elif key in BINARY:
                lchildren = input_ids[key][0]
                rchildren = input_ids[key][1]
                assert(len(lchildren) == len(rchildren) and len(lchildren) > 0)
                lhide = torch.stack([embeddings[child_id][0] for child_id in lchildren], dim=0)
                lcell = torch.stack([embeddings[child_id][1] for child_id in lchildren], dim=0)
                rhide = torch.stack([embeddings[child_id][0] for child_id in rchildren], dim=0)
                rcell = torch.stack([embeddings[child_id][1] for child_id in rchildren], dim=0)
                inputs[key] = [(lhide, lcell), (rhide, rcell)]
            else:
                AssertionError("[%s] is not a valid block", key)
            ids[key] = [parents[idx] for idx in input_ids[key][0]]
        return inputs, ids

    # Given batch of binary trees with computed embeddings, returns tensor of size [maxlen, batch_size, model_dim] by flattening each tree
    def flatten_tree(self, embeddings, id2emb, maxlen):        
        id2emb = [tree + [tree[0]] + [0] * (maxlen-1-len(tree)) for tree in id2emb] # Use maxlen-1 because we already account for the end padding
        zeros = torch.zeros(self.dim, dtype=torch.float)
        zeros = zeros if self.cpu else zeros.cuda()
        embeddings[0] = zeros, 0
        return torch.stack([torch.stack([embeddings[idx][0] for idx in tree], dim=0) for tree in id2emb], dim=0)

    # Returns the embedding of an integer using its binary embedding
    def int_emb(self, integer):
        b = bin(integer)[2:]
        if len(b) > self.num_bit:
            return "Number too large"
        place = torch.LongTensor([i for i in range(len(b)) if b[len(b)-i-1] == '1'])
        place = place if self.cpu else place.cuda()
        return torch.sum(self.bin2emb(place), 0)

    # Helper function for labeling batch. Calls label_and_map_tree() for the BinEqnTree object.
    def label_and_map_batch(self, batch):
        max_depth = max([tree.depth for tree in batch]) + 1 # Add 1 to account for additional leaves
        unused_id = [1]
        root = []
        children = {}
        parents = {}
        embedding = {}
        def child_ids():
            return [[], []]
        buckets = [defaultdict(child_ids) for i in range(max_depth + 1)]
        labels = []
        depths = []
        batch_size = 0
        for tree in batch:
            current_depth = tree.depth + 1
            tree_children = {}
            tree_parents = {}
            tree_embedding = {}
            root.append(unused_id[0])
            tree.label_and_map_tree(unused_id, current_depth, tree_children, tree_parents, tree_embedding, buckets)
            children.update(tree_children)
            parents.update(tree_parents)
            embedding.update(tree_embedding)      
            batch_size += 1
        return children, parents, embedding, buckets, max_depth, batch_size, root

    # Given a list of tokens in prefix form, convert to a BinaryEqnTree object for decoding.
    def prefix_to_tree(self, token_list):
        trees = []
        id2embs = []
        lengths = []
        valid_idx = []
        counter = 1
        for i in range(len(token_list)):
            old_counter = counter # reset the counter in case we come across an invalid equation
            tree, id2emb, _, counter, valid = self._prefix_to_tree(token_list[i], [], counter=counter)
            if valid:
                trees.append(tree)
                id2embs.append(id2emb)
                lengths.append(len(id2emb)+1) # Add 1 to account for the end padding
                valid_idx.append(i)
            else:
                counter = old_counter
        return trees, id2embs, torch.LongTensor(lengths), valid_idx

    def _prefix_to_tree(self, tokens, id2emb, idx=0, counter=1):
        token = tokens[idx]
        idx += 1
        id2emb.append(counter)
        counter += 1
        valid = True

        if token in BINARY:
            left, _, idx, counter, valid1 = self._prefix_to_tree(tokens, id2emb, idx=idx, counter=counter)
            right, _, idx, counter, valid2 = self._prefix_to_tree(tokens, id2emb, idx=idx, counter=counter) 
            valid = valid1 and valid2
            root = BinaryEqnTree(token, left, right)
        elif token == EOS or token in UNARY:
            left, _, idx, counter, valid = self._prefix_to_tree(tokens, id2emb, idx=idx, counter=counter)
            root = BinaryEqnTree(token, left, None)
        elif token in INT:
            counter += 1
            val = ""
            while idx < len(tokens) and tokens[idx] in DIGITS:
                val += tokens[idx]
                idx += 1
            if int(val) >= (1 << self.num_bit):
                valid = False
            root = BinaryEqnTree(token, BinaryEqnTree(val, None, None, value=self.int_emb(int(val))), None)
        elif token in DERIVATIVES:
            counter += 2
            value = torch.LongTensor([VOCAB['Y']])[0]
            value = value if self.cpu else value.cuda()
            root = BinaryEqnTree(SYMBOL_ENCODER, BinaryEqnTree('Y', None, None, value=value), None)
            root = BinaryEqnTree('d'+str(len(token)-1), root, None)
        elif token in LEAF:
            counter += 1
            value = torch.LongTensor([VOCAB[token]])[0]
            value = value if self.cpu else value.cuda()
            root = BinaryEqnTree(SYMBOL_ENCODER, BinaryEqnTree(token, None, None, value=value), None)
        else:
            raise AssertionError("{0} is not a valid symbol".format(token))

        root.depth = root.get_depth()
        return root, id2emb, idx, counter, valid


class BinaryLSTMNodeSym(torch.nn.Module):

    def __init__(self, num_input, num_hidden):
        super().__init__()
        self.data = nn.Linear(num_input, num_hidden, bias=False)

        self.data_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.forget_by_self = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_by_opposite = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.output = nn.Linear(num_input, num_hidden, bias=False)
        self.output_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.input = nn.Linear(num_input, num_hidden, bias=False)
        self.input_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))

    def forward(self, input_left, input_right, dropout=None):
        """

        Args:
            input_left: ((num_hidden,), (num_hidden,))
            input_right: ((num_hidden,), (num_hidden,))

        Returns:
            (num_hidden,), (num_hidden)
        """
        hl, cl = input_left
        hr, cr = input_right
        i = torch.sigmoid(self.data(hl) + self.data(hr) + self.data_bias)
        f_left = torch.sigmoid(self.forget_by_self(hl) +
                           self.forget_by_opposite(
                               hr) + self.forget_bias)
        f_right = torch.sigmoid(self.forget_by_opposite(hl) +
                           self.forget_by_self(
                               hr) + self.forget_bias)
        o = torch.sigmoid(self.output(hl) + self.output(hr) + self.output_bias)
        u = torch.tanh(self.input(hl) + self.input(hr) + self.input_bias)
        if dropout is None:
            c = i * u + f_left * cl + f_right * cr
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f_left * cl + f_right * cr
        h = o * torch.tanh(c)
        return h, c


class BinaryLSTMNode(torch.nn.Module):

    def __init__(self, num_input, num_hidden):
        super().__init__()
        self.data_left = nn.Linear(num_input, num_hidden, bias=False)
        self.data_right = nn.Linear(num_input, num_hidden, bias=False)
        self.data_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.forget_left_by_left = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_left_by_right = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_right_by_left = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_right_by_right = nn.Linear(num_input, num_hidden, bias=False)
        self.forget_bias_left = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.forget_bias_right = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.output_left = nn.Linear(num_input, num_hidden, bias=False)
        self.output_right = nn.Linear(num_input, num_hidden, bias=False)
        self.output_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))
        self.input_left = nn.Linear(num_input, num_hidden, bias=False)
        self.input_right = nn.Linear(num_input, num_hidden, bias=False)
        self.input_bias = nn.Parameter(torch.FloatTensor([0] * num_hidden))

    def forward(self, input_left, input_right, dropout=None):
        """

        Args:
            input_left: ((num_hidden,), (num_hidden,))
            input_right: ((num_hidden,), (num_hidden,))

        Returns:
            (num_hidden,), (num_hidden)
        """
        hl, cl = input_left
        hr, cr = input_right
        i = torch.sigmoid(self.data_left(hl) + self.data_right(hr) + self.data_bias)
        f_left = torch.sigmoid(self.forget_left_by_left(hl) +
                           self.forget_left_by_right(
                               hr) + self.forget_bias_left)
        f_right = torch.sigmoid(self.forget_right_by_left(hl) +
                           self.forget_right_by_right(
                               hr) + self.forget_bias_right)
        o = torch.sigmoid(self.output_left(hl) + self.output_right(hr) + self.output_bias)
        u = torch.tanh(self.input_left(hl) + self.input_right(hr) + self.input_bias)
        if dropout is None:
            c = i * u + f_left * cl + f_right * cr
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f_left * cl + f_right * cr
        h = o * torch.tanh(c)
        return h, c


class UnaryLSTMNode(torch.nn.Module):
    def __init__(self, num_input, num_hidden):
        super().__init__()
        self.data = nn.Linear(num_input, num_hidden, bias=True)
        self.forget = nn.Linear(num_input, num_hidden, bias=True)
        self.output = nn.Linear(num_input, num_hidden, bias=True)
        self.input = nn.Linear(num_input, num_hidden, bias=True)

    def forward(self, inp, dropout=None):
        """

        Args:
            inp: ((num_hidden,), (num_hidden,))

        Returns:
            (num_hidden,), (num_hidden)
        """
        h, c = inp
        i = torch.sigmoid(self.data(h))
        f = torch.sigmoid(self.forget(h))
        o = torch.sigmoid(self.output(h))
        u = torch.tanh(self.input(h))
        if dropout is None:
            c = i * u + f * c
        else:
            c = i * F.dropout(u,p=dropout,training=self.training) + f * c
        h = o * torch.tanh(c)
        return h, c
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


class TreeLSTM_Encoder(torch.nn.Module):
    def __init__(self, params, id2word, word2id, una_ops, bin_ops):
        super().__init__()

        self.d_model = params.emb_dim
        self.id2word = id2word
        self.word2id = word2id
        self.una_ops = una_ops
        self.bin_ops = bin_ops
        self.pad_idx = -1
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

    '''
    operations: torch.Tensor,
    tokens: torch.Tensor,
    left_idx: torch.Tensor,
    right_idx: torch.Tensor,
    depths: torch.Tensor,
    operation_order: torch.Tensor,
    digits: torch.Tensor
    '''
    def forward(
        self, x=None, lengths=None,
    ) -> torch.Tensor:
        """
        Given a batch of tensorized trees, produce the model log probability that
        equality holds for each.
        """
        operations, tokens, left_idx, right_idx, depths, operation_order, _ = x
        num_steps = operation_order.numel()
        num_nodes = operations.numel()
        activations = torch.zeros(
            (num_nodes, 2, self.d_model), dtype=torch.float, device=operations.device
        )

        for depth in range(num_steps):  # type: ignore
            step_mask = depths == depth  # Indices to compute at this step
            op = operation_order[depth].item()

            if op in [-1, -2]:  # Embedding lookup or number encoding
                idx = tokens.masked_select(step_mask)
                step_activations = self.leaf_emb(idx) if op == -1 else self.num_enc(idx.float().unsqueeze(1))
                zeros = torch.zeros(
                    (len(idx), self.d_model), dtype=torch.float, device=operations.device
                )
                step_activations = torch.stack([step_activations, zeros], dim=1)
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

        hidden, _ = torch.split(activations, 1, dim=1)
        '''
        if digits > 1:
            zeros = torch.zeros(
                (num_nodes, digits-1, self.d_model), dtype=torch.float, device=hidden.device
            )
            hidden_pad = torch.cat((hidden, zeros), dim=0)
        else:
            hidden_pad = hidden
        hidden_pad = torch.where(int_mask.unsqueeze(1).unsqueeze(1), int_embs, hidden_pad)
        dim = hidden_pad.size()
        hidden_pad_flat = hidden_pad.view(dim[0]*dim[1], self.d_model)
        nonzero = (hidden_pad_flat != 0).all(1)
        hidden = hidden_pad_flat[nonzero, :]
        '''
        unpadded_batch = torch.split(hidden, lengths.tolist(), dim=0)
        return pad_sequence(unpadded_batch, padding_value=0.0, batch_first=True).squeeze(2)
