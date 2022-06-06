import math
import torch
import torch.nn.functional as F
from torch import nn


# Not sure yet if this needs the style embedding to be separated during pretrain.
class Transformer(nn.Module):
    def __init__(self, params, vocab, cxt_vocab=None, config=None):
        super(Transformer, self).__init__()
        num_layers = params.num_layers
        h, dropout = params.h, params.dropout
        d_model, max_length = params.d_model, params.max_length
        self.max_length = max_length
        self.d_model = d_model
        self.vocab_size = len(vocab)
        self.eos_idx = vocab.stoi['<eos>']
        self.pad_idx = vocab.stoi['<pad>']

        self.config = config

        self.embed = EmbeddingLayer(len(vocab), d_model, self.max_length)
        # start of sequence token
        self.sos_token = nn.Parameter(torch.randn(d_model))
        # Encoder and decoder share vocab
        self.encoder = Encoder(num_layers, d_model, h, dropout)
        self.decoder = Decoder(num_layers, d_model, len(
            vocab), h, dropout, self.embed.token_embed.weight)

        self.cxt_vocab = cxt_vocab
        if self.config is not None and 'emb' in self.config:
            assert cxt_vocab is not None
            self.coher_emb = Embedding(len(cxt_vocab), d_model)

    def expand_embeddings(self, new_vocab_size, device):
        old_embeddings = self.embed.token_embed.weight.data.clone().to(device)
        old_vocab_size = old_embeddings.size(0)

        assert new_vocab_size >= old_vocab_size

        # Reinitialise embedding weights
        self.embed.token_embed = Embedding(new_vocab_size, self.d_model).to(device)
        self.embed.token_embed.weight.data[:old_vocab_size, :] = old_embeddings

        # Tie softmax weights to embedding weights
        self.decoder.generator = Generator(self.d_model, new_vocab_size, self.embed.token_embed.weight)

        return

    def expand_pos_embeddings(self, new_size, device):
        old_embeddings = self.embed.pos_embed.weight.data.clone().to(device)
        self.embed.pos_embed = Embedding(new_size, self.d_model).to(device)

        assert new_size >= old_embeddings.size(0)

        self.embed.pos_embed.weight.data[:old_embeddings.size(0), :] = old_embeddings
        return

    def add_coher_embedding(self, inp_len, device):
        self.coher_emb = Embedding(inp_len, self.d_model).to(device)

    def add_bias_embedding(self, inp_len, device):
        self.decoder.generator.coher_emb = Embedding(inp_len, self.vocab_size).to(device)

    def pad_hyps(self, t, padding_len):
        pad_t = torch.Tensor(padding_len).fill_(self.pad_idx)
        pad_t[:t.size(0)] = t
        t = pad_t.to(torch.int64)[1:]
        return t

    def pad(self, t, padding_len):
        pad_t = torch.Tensor(padding_len, 1).fill_(-2e9)
        pad_t[:t.size(0)] = t
        return pad_t

    def beam_search_decoder(self, pos_idx, inp_lengths, max_enc_len, src_mask, trg_mask, memory, device,
                            sos_token, batch_size, beam_size=1, tag_dec=False, target_types=None):
        # Beam search decoding
        bk_size = batch_size * beam_size

        # First prediction step
        log_prob, prev_states = self.decoder.incremental_forward(
            sos_token, memory,
            src_mask, trg_mask[:, :, 0:1, :1]
        )

        # This is going to be <sos>
        first_token = self.embed(log_prob.argmax(-1), pos_idx[:, :1])
        hypotheses = log_prob.argmax(-1).to(device)
        # Generating from <sos>
        second_token_log_prob, prev_states = self.decoder.incremental_forward(
            first_token, memory,
            src_mask, trg_mask[:, :, 1:2, :2],
            prev_states
        )

        # Pick best k from the first prediction.
        top_prob, top_idx = second_token_log_prob.topk(beam_size, sorted=True)

        # Copy prev_states across k beams
        for l in range(len(prev_states) - 1):
            for s in range(len(prev_states[l])):
                prev_states[l][s] = prev_states[l][s].repeat(beam_size, 1, 1)
        prev_states[-1] = prev_states[-1].repeat(beam_size, 1, 1)

        # Expand batch size to meet the beam search's needs
        pos_idx = pos_idx.repeat((beam_size, 1))
        inp_lengths = inp_lengths.repeat(beam_size)
        memory = memory.repeat((beam_size, 1, 1))
        if self.config == 'out_bias':
            self.decoder.generator.types = self.decoder.generator.types.repeat((beam_size, 1))
        src_mask = pos_idx[:, :max_enc_len] >= inp_lengths.unsqueeze(-1)

        if self.config in ['emb_enc', 'emb_enc_dec'] and target_types is not None:
            src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), 1)
            src_mask = src_mask.view(bk_size, 1, 1, max_enc_len + 1)
        else:
            src_mask = src_mask.view(bk_size, 1, 1, max_enc_len)

        # Make next token & initialise a buffer for previous tokens
        if tag_dec and target_types is not None:
            target_types = target_types.transpose(1, 0)
            top_idx = target_types[0, :].unsqueeze(1).repeat((1, 1, beam_size))
            top_prob = log_prob[range(top_idx.shape[0]), :, top_idx]
        top_prob = top_prob.transpose(0, 2).reshape(-1).unsqueeze(-1)
        top_idx = top_idx.transpose(0, 2).reshape(-1).unsqueeze(-1)

        next_token = self.embed(top_idx, pos_idx[:, 1:2])
        hypotheses = torch.cat((hypotheses.repeat((beam_size, 1)), top_idx), 1)
        accum_log_prob = top_prob
        k_ = 2
        if tag_dec and target_types is not None:
            target_types = target_types.repeat((1, beam_size))
            # Set to -1 so that we do not predict null; let the model predict it on its own
            for type_ in target_types[1:-1, :]:
                log_prob, prev_states = self.decoder.incremental_forward(
                    next_token, memory,
                    src_mask, trg_mask[:, :, k_:k_ + 1, :k_ + 1],
                    prev_states
                )
                hypotheses = torch.cat((hypotheses, type_.unsqueeze(-1)), 1)
                accum_log_prob += log_prob[range(type_.shape[0]), :, type_]
                next_token = self.embed(type_.unsqueeze(-1), pos_idx[:, 1:2])
                k_ += 1
        padding_len = self.max_length
        # safe deposit boxes for ended sequences
        safe_hypotheses = [[] for _ in range(batch_size)]
        safe_hypotheses_log_probs = [[] for _ in range(batch_size)]
        top_hypotheses = ['' for _ in range(batch_size)]
        for k in range(k_, self.max_length):
            # Generate the log probs of the next token
            log_prob, prev_states = self.decoder.incremental_forward(
                next_token, memory,
                src_mask, trg_mask[:, :, k:k + 1, :k + 1],
                prev_states
            )

            t = k
            # hyp_log_prob [128, 1, 32004] contains hypothesis probabilities
            # Probs of hypotheses = probs accumulated so far + probs for next token. Don't add if hypothesis is finished
            hyp_log_prob = accum_log_prob.unsqueeze(-1) + log_prob
            # Pick top beam_size candidates within every beam
            # Shape of log prob is [batch, batch, batch, batch]
            top = torch.stack((hyp_log_prob.topk(beam_size, sorted=True)), 3)
            # stack batches : top 4 from beam 0, 1, 2 and 3 are within one batch. copied 4 times
            y = top.reshape(beam_size, batch_size, beam_size, -1).transpose(0, 1).reshape(batch_size,
                                                                                          beam_size * beam_size,
                                                                                          -1).repeat(beam_size, 1, 1)
            # Pick top 4 tokens within every sample
            val, ind = (y[:, :, 0]).topk(k=beam_size, sorted=True)
            # Create a mask for beams
            ints = (torch.Tensor(range(bk_size)) /
                    batch_size).unsqueeze(-1).to(torch.int64).to(device)
            # Topk_spread is topk but every beam gets its own 1st, 2nd, 3rd best element
            topk_spread = ind.gather(1, ints.expand_as(ind))[:, 0]
            # next_tokens is takes indices arrived at above and collects values from y for each sample
            next_tokens = y[:, :, 1].gather(
                1, topk_spread.unsqueeze(1)).to(torch.int64)
            best_hyp_probs = y[:, :, 0].gather(1, topk_spread.unsqueeze(1))

            # Now that we have the next token we must recover the hypothesis
            # Which beam did I come from?
            original_pos = ((topk_spread / beam_size).to(torch.int64))
            original_pos = (
                    torch.Tensor(range(batch_size)).repeat(beam_size).to(device) + original_pos * batch_size).to(
                torch.int64)

            # Reorder previous hypotheses:
            # 1. Update hypotheses
            hypotheses = hypotheses[original_pos, :]
            # 2. Update prev_states
            for l in range(len(prev_states) - 1):
                for s in range(len(prev_states[l])):
                    prev_states[l][s] = prev_states[l][s][original_pos, :, :]
            prev_states[-1] = prev_states[-1][original_pos, :, :]

            # If we're at the end then add all hypotheses from beam 1
            if k + 1 == self.max_length:
                for i in range(batch_size):
                    safe_hypotheses[i].append(
                        self.pad_hyps(hypotheses[i], padding_len))
                    safe_hypotheses_log_probs[i].append(best_hyp_probs[i] / t)
            # if we're not at the end yet, then just those that reached EOS
            else:
                done_mask = next_tokens.squeeze() == self.eos_idx
                ended = done_mask.nonzero(as_tuple=False)
                accum_log_prob = best_hyp_probs + \
                                 (done_mask * -2e9).unsqueeze(-1)
                hypotheses = torch.cat((hypotheses, next_tokens), 1)
                if ended.size(0):
                    for v in ended:
                        v, sample_number = int(v), int(v % batch_size)
                        safe_hypotheses[sample_number].append(
                            self.pad_hyps(hypotheses[v], padding_len))
                        safe_hypotheses_log_probs[sample_number].append(
                            best_hyp_probs[v] / t)
                next_token = self.embed(next_tokens, pos_idx[:, k:k + 1])

        padding_len = max([len(s) for s in safe_hypotheses_log_probs])
        safe_hypotheses_log_probs = [self.pad(torch.stack(
            s), padding_len) for s in safe_hypotheses_log_probs]
        safe_hypotheses_log_probs = torch.cat(
            safe_hypotheses_log_probs, 1).transpose(1, 0)

        safe_hypotheses = [torch.stack(s) for s in safe_hypotheses]
        _, top_vals = safe_hypotheses_log_probs.topk(1, sorted=True)
        top_vals = top_vals.squeeze(-1).tolist()
        for i in range(len(safe_hypotheses)):
            top_hypotheses[i] = safe_hypotheses[i][top_vals[i]]
        top_hypotheses = torch.stack(top_hypotheses).to(device)
        return top_hypotheses

    def forward(self, inp_tokens, gold_tokens, inp_lengths, types, generate=False, beam_size=1, tag_dec=False):
        assert types is not None
        assert torch.equal(self.decoder.generator.proj.weight.data, self.embed.token_embed.weight.data), "Tied weights are different."
        batch_size = inp_tokens.size(0)
        max_enc_len = inp_tokens.size(1)

        assert max_enc_len <= self.max_length

        pos_idx = torch.arange(self.max_length).unsqueeze(0).expand((batch_size, -1))
        pos_idx = pos_idx.to(inp_lengths.device)

        src_mask = pos_idx[:, :max_enc_len] >= inp_lengths.unsqueeze(-1)

        if self.config == 'out_bias' and types is not None:
            # Passing information to the generator
            self.decoder.generator.types = types

        if self.config in ['emb_enc', 'emb_enc_dec'] and types is not None:
            # Prepending zeros to the mask
            src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), 1)
            src_mask = src_mask.view(batch_size, 1, 1, max_enc_len + 1)
        else:
            src_mask = src_mask.view(batch_size, 1, 1, max_enc_len)
        trg_mask = torch.ones(
            (self.max_length, self.max_length)).to(src_mask.device)
        trg_mask = (trg_mask.tril() == 0).view(1, 1, self.max_length, self.max_length)

        if 'emb' in self.config:
            coher_emb = torch.mean(self.coher_emb(types), dim=1, keepdim=True)

        if self.config in ['emb_enc', 'emb_enc_dec'] and types is not None:
            enc_input = torch.cat((coher_emb, self.embed(inp_tokens, pos_idx[:, :max_enc_len])), 1)

        elif self.config == 'emb_pw_sum' and types is not None:
            # Element-wise sum of coherence embedding and source words embedding.
            enc_input = self.embed(inp_tokens, pos_idx[:, :max_enc_len]) + coher_emb.repeat((1, max_enc_len, 1))

        else:
            enc_input = self.embed(inp_tokens, pos_idx[:, :max_enc_len])
        enc_output = self.encoder(enc_input, src_mask)

        if self.config == 'emb_add':
            enc_output = enc_output + coher_emb.expand_as(enc_output)

        if self.config in ['emb_dec', 'emb_enc_dec'] and types is not None:
            # sos_token becomes the mean of attributes
            sos_token = coher_emb
        else:
            sos_token = self.sos_token.view(1, 1, -1).expand(batch_size, -1, -1)
        if not generate:
            max_dec_len = gold_tokens.size(1)
            dec_input = gold_tokens[:, :-1]
            dec_input_emb = torch.cat((sos_token, self.embed(dec_input, pos_idx[:, :max_dec_len - 1])), 1)

            log_probs = self.decoder(
                dec_input_emb, enc_output,
                src_mask, trg_mask[:, :, :max_dec_len, :max_dec_len],
            )
            return log_probs

        elif generate:
            return self.beam_search_decoder(pos_idx, inp_lengths, max_enc_len, src_mask, trg_mask, enc_output,
                                            inp_tokens.device,
                                            sos_token, batch_size, beam_size=beam_size, tag_dec=tag_dec,
                                            target_types=types)


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, h, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        y = x
        assert y.size(1) == mask.size(-1)

        for layer in self.layers:
            y = layer(y, mask)
        return self.norm(y)


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, vocab_size, h, dropout, tied_weights):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
        # Want generator to be the transpose of the embedding matrix
        self.generator = Generator(d_model, vocab_size, tied_weights)

    def forward(self, x, memory, src_mask, trg_mask):
        y = x
        assert y.size(1) == trg_mask.size(-1)
        for layer in self.layers:
            y = layer(y, memory, src_mask, trg_mask)
        return self.generator(self.norm(y))

    def incremental_forward(self, x, memory, src_mask, trg_mask, prev_states=None):
        y = x

        new_states = []
        for i, layer in enumerate(self.layers):
            y, new_sub_states = layer.incremental_forward(
                y, memory, src_mask, trg_mask,
                prev_states[i] if prev_states else None
            )
            new_states.append(new_sub_states)

        new_states.append(
            torch.cat((prev_states[-1], y), 1) if prev_states else y)
        y = self.norm(new_states[-1])[:, -1:]

        return self.generator(y), new_states


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size, tied_weights):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)  # Bias set to false because we're tying weights.
        self.proj.weight = tied_weights
        self.config = None

    def forward(self, x):
        if self.config == 'out_bias':
            coherence_bias = torch.mean(self.coher_emb(self.types), dim=1, keepdim=True)
            return F.log_softmax(self.proj(x) + coherence_bias, dim=-1)
        return F.log_softmax(self.proj(x), dim=-1)


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, max_length):
        super(EmbeddingLayer, self).__init__()
        self.token_embed = Embedding(vocab_size, d_model)
        self.pos_embed = Embedding(max_length, d_model)
        self.vocab_size = vocab_size

    def forward(self, x, pos):
        if len(x.size()) == 2:
            y = self.token_embed(x) + self.pos_embed(pos)
        else:
            y = torch.matmul(x, self.token_embed.weight) + self.pos_embed(pos)

        return y


class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
        self.sublayer = nn.ModuleList(
            [SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.pw_ffn)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.src_attn = MultiHeadAttention(d_model, h, dropout)
        self.pw_ffn = PositionwiseFeedForward(d_model, dropout)
        self.sublayer = nn.ModuleList(
            [SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, trg_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, trg_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.pw_ffn)

    def incremental_forward(self, x, memory, src_mask, trg_mask, prev_states=None):
        new_states = []
        m = memory
        x = torch.cat((prev_states[0], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[0].incremental_forward(
            x, lambda x: self.self_attn(x[:, -1:], x, x, trg_mask))
        x = torch.cat((prev_states[1], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[1].incremental_forward(
            x, lambda x: self.src_attn(x[:, -1:], m, m, src_mask))
        x = torch.cat((prev_states[2], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[2].incremental_forward(
            x, lambda x: self.pw_ffn(x[:, -1:]))
        return x, new_states

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.head_projs = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)])
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for x, l in zip((query, key, value), self.head_projs)]

        attn_feature, _ = scaled_attention(query, key, value, mask)
        attn_concated = attn_feature.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.fc(attn_concated)


def scaled_attention(query, key, value, mask):
    d_k = query.size(-1)
    scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
    scores.masked_fill_(mask, float('-inf'))
    attn_weight = F.softmax(scores, -1)
    attn_feature = attn_weight.matmul(value)
    return attn_feature, attn_weight


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.mlp = nn.Sequential(
            Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.mlp(x)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x + self.dropout(y)

    def incremental_forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x[:, -1:] + self.dropout(y)


def Linear(in_features, out_features, bias=True, uniform=True):
    """    Creates a linear layer & initialises weights with Xavier uniform.
    Torch Linear INVERTS the tensor dimensions when created, i.e. Linear(in, out) is a tensor(out, in)
    """
    m = nn.Linear(in_features, out_features, bias)
    if uniform:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.xavier_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.xavier_uniform_(m.weight)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim, eps=1e-6):
    m = nn.LayerNorm(embedding_dim, eps)
    return m
