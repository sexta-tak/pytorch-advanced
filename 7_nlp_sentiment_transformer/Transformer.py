#%%
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

#%%
class Embedder(nn.Module):
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(embeddings=text_embedding_vectors, freeze=True)

    def forward(self, x):
        x_vec = self.embeddings(x)

        return x_vec

class PositionalEncoder(nn.Module):
    def __init__(self, d_model=300, max_seq_len=256):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**((2*i)/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i)/d_model)))

        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, x):
        return x + self.pe

class Attention(nn.Module):
    def __init__(self, d_model=300):
        super().__init__()

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linaer = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)
        self.d_k = d_model**0.5

    def forward(self, x, mask):
        k = self.k_linaer(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        weights = torch.matmul(q, k.transpose(1, 2)) / self.d_k
        mask = mask.unsqueeze(1)
        normalized_weights = F.softmax(weights, dim=-1)

        output = torch.matmul(normalized_weights, v)

        return output, normalized_weights

class FeedForward(nn.Modlue):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linaer_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linaer_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = Attention(d_model)
        self.ff = FeedForward(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_normalized = self.norm1(x)
        output, normlized_weights = self.attn(x_normalized, mask)

        x2 = x + self.dropout1(output)
        x_normalized2 = self.norm2(x2)
        output = x2 + self.dropout2(self.ff(x_normalized2))

        return output, normlized_weights

class ClassificationHead(nn.Module):
    def __init__(self, d_model=300, output_dim=2):
        super().__init__()

        self.linear = nn.Linear(d_model, output_dim)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        x0 = x[:, 0, :]
        out = self.linear(x0)

        return out

class TransformerClassification(nn.Module):
    def __init__(self, text_embedding_vectors, d_model=300, max_seq_len=256, output_dim=2):
        super().__init__()

        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(d_moodel=d_model, max_seq_len=max_seq_len)
        self.net3_1 = TransformerBlock(d_model=d_model)
        self.net3_2 = TransformerBlock(d_model=d_model)
        self.net4 = ClassificationHead(output_dim=output_dim, d_model=d_model)

    def forward(self, x, mask):
        x1 = self.net(x)
        x2 = self.net2(x1)
        x3_1, normalized_weights1 = self.net3_1(x2, mask)
        x3_2, normalized_weights2 = self.net3_2(x3_1, mask)
        x4 = self.net4(x3_2)

        return x4, normalized_weights1, normalized_weights2