import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder

from .utils import init_param, Scaler


def spectral_init(weight, rank):

    U, S, V = torch.svd(weight)
    sqrtS = torch.diag(torch.sqrt(S[:rank]))
    return torch.matmul(U[:,:rank], sqrtS), torch.matmul(V[:,:rank], sqrtS).T

class FactorizedLinear(nn.Module):

    def __init__(self, embedding_size, hidden_size, LR_ratio=1.0, init='spectral'):

        super(FactorizedLinear, self).__init__()
        self.embedding_size = embedding_size
        self.rank_embedding_size = int(round(min(embedding_size, hidden_size) * LR_ratio))

        original_linear = nn.Linear(embedding_size, hidden_size, bias=False)
        original_linear.weight.data.uniform_(-0.1, 0.1)
        # spectral initialization
        U, VT = spectral_init(original_linear.weight, self.rank_embedding_size)

        modules = [nn.Linear(self.embedding_size, self.rank_embedding_size, bias=False)]
        modules.append(nn.Linear(self.rank_embedding_size, hidden_size, bias=False))

        modules[0].weight.data = VT
        modules[1].weight.data = U
        self.linear = nn.Sequential(*modules)

    def forward(self, x):
        return self.linear(x)

    def init_param(self):
        for module in self.linear:
            module.weight.data.normal_(mean=0.0, std=0.02)


class FactorizedEmbedding(nn.Module):

    def __init__(self, num_tokens, embedding_size, dropout, rate, LR_ratio, cfg):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.rank_embedding_size = int(round(min(embedding_size, num_tokens + 1) * LR_ratio))

        original_embedding = nn.Embedding(num_tokens + 1, embedding_size)
        original_embedding.weight.data.normal_(0, 0.1)
        # spectral initialization
        U, VT = spectral_init(original_embedding.weight, self.rank_embedding_size)
        modules = [nn.Embedding(num_tokens + 1, self.rank_embedding_size)]
        modules.append(nn.Linear(self.rank_embedding_size, embedding_size, bias=False))
        modules[0].weight.data = U
        modules[1].weight.data = VT.T
        self.embedding = nn.Sequential(*modules)
        self.positional_embedding = FactorizedPositionalEmbedding(embedding_size, LR_ratio, cfg)
        self.norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.scaler = Scaler(rate)

    def forward(self, src):
        src = self.scaler(self.embedding(src)) + self.scaler(self.positional_embedding(src))
        src = self.dropout(self.norm(src))
        return src


class FactorizedPositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, LR_ratio, cfg):
        super().__init__()
        self.rank_embedding_size = int(round(min(embedding_size, cfg['bptt'] + 1) * LR_ratio))

        original_embedding = nn.Embedding(cfg['bptt'], embedding_size)
        original_embedding.weight.data.normal_(0, 0.1)
        # spectral initialization
        U, VT = spectral_init(original_embedding.weight, self.rank_embedding_size)

        modules = [nn.Embedding(cfg['bptt'], self.rank_embedding_size)]
        modules.append(nn.Linear(self.rank_embedding_size, embedding_size, bias=False))
        modules[0].weight.data = U
        modules[1].weight.data = VT.T
        self.positional_embedding = nn.Sequential(*modules)

        # self.positional_embedding = nn.Embedding(cfg['bptt'], embedding_size)

    def forward(self, x):
        N, S = x.size()
        position = torch.arange(S, dtype=torch.long, device=x.device).unsqueeze(0).expand((N, S))
        x = self.positional_embedding(position)
        return x

class ScaledDotProduct(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        scores = q.matmul(k.transpose(-2, -1)) / self.temperature
        seq_len = scores.shape[-1]
        h = scores.shape[0]
        mask = torch.tril(torch.ones((h, seq_len, seq_len))).to(str(scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, rate):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.linear_q = nn.Linear(embedding_size, embedding_size)
        self.linear_k = nn.Linear(embedding_size, embedding_size)
        self.linear_v = nn.Linear(embedding_size, embedding_size)
        self.linear_o = nn.Linear(embedding_size, embedding_size)
        self.attention = ScaledDotProduct(temperature=(embedding_size // num_heads) ** 0.5)
        self.scaler = Scaler(rate)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim).permute(0, 2, 1, 3) \
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature).permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.scaler(self.linear_q(q)), self.scaler(self.linear_k(k)), self.scaler(self.linear_v(v))
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        q, attn = self.attention(q, k, v, mask)
        q = self._reshape_from_batches(q)
        q = self.scaler(self.linear_o(q))
        return q, attn


class FactorizedTransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout, rate, LR_ratio):
        super().__init__()
        self.mha = MultiheadAttention(embedding_size, num_heads, rate=rate)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.linear1 = FactorizedLinear(embedding_size, hidden_size, LR_ratio)
        # self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = FactorizedLinear(hidden_size, embedding_size, LR_ratio)
        # self.linear2 = nn.Linear(hidden_size, embedding_size)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.scaler = Scaler(rate)
        self.activation = nn.GELU()
        self.init_param()

    def init_param(self):
        # self.linear1.init_param()
        # self.linear2.init_param()
        # self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        # self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.weight.data.fill_(1.0)
        self.norm1.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        return

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        attn_output, _ = self.mha(src, src, src, mask=src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        src2 = self.scaler(self.linear2(self.dropout1(self.activation(self.scaler(self.linear1(src))))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class FactorizedDecoder(nn.Module):
    def __init__(self, num_tokens, embedding_size, rate, LR_ratio):
        super().__init__()
        self.linear1 = FactorizedLinear(embedding_size, embedding_size, LR_ratio)
        # self.linear1 = nn.Linear(embedding_size, embedding_size)
        self.scaler = Scaler(rate)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.linear2 = FactorizedLinear(embedding_size, num_tokens, LR_ratio)
        # self.linear2 = nn.Linear(embedding_size, num_tokens)

    def forward(self, src):
        out = self.linear2(self.norm1(self.activation(self.scaler(self.linear1(src)))))
        return out


class HyperTransformer(nn.Module):
    def __init__(self, num_tokens, embedding_size, num_heads, hidden_size, num_layers, dropout, rate, LR_ratio, cfg):
        super().__init__()
        self.num_tokens = num_tokens
        # self.transformer_embedding = TransformerEmbedding(num_tokens, embedding_size, dropout, rate, cfg)
        self.transformer_embedding = FactorizedEmbedding(num_tokens, embedding_size, dropout, rate, LR_ratio, cfg)
        encoder_layers = FactorizedTransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout, rate, LR_ratio)
        # encoder_layers = TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout, rate)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.decoder = FactorizedDecoder(num_tokens, embedding_size, rate, LR_ratio)
        # self.decoder = Decoder(num_tokens, embedding_size, rate)
        self.cfg = cfg

    def forward(self, input):

        src = input.clone()
        src = src.unsqueeze(0)
        N, S = src.size()
        d = torch.distributions.bernoulli.Bernoulli(probs=self.cfg['mask_rate'])
        mask = d.sample((N, S)).to(src.device)
        src = src.masked_fill(mask == 1, self.num_tokens).detach()
        src = self.transformer_embedding(src)
        src = self.transformer_encoder(src)
        out = self.decoder(src)
        out = out.permute(0, 2, 1)
        return out


def hyper_transformer(model_rate=1, ratio_LR=1, cfg=None):
    num_tokens = cfg['num_tokens']
    embedding_size = int(np.ceil(model_rate * cfg['transformer']['embedding_size']))
    num_heads = cfg['transformer']['num_heads']
    hidden_size = int(np.ceil(model_rate * cfg['transformer']['hidden_size']))
    num_layers = cfg['transformer']['num_layers']
    dropout = cfg['transformer']['dropout']
    scaler_rate = model_rate / cfg['global_model_rate']
    model = HyperTransformer(num_tokens, embedding_size, num_heads, hidden_size, num_layers, dropout, scaler_rate, ratio_LR, cfg)
    return model
