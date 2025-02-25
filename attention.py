import torch
import numpy as np
from torch import nn
import math

class ScaledDotProductAttension(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, dim=32):
        super().__init__()
        self.scale = math.sqrt(dim)
        self.softmax = nn.Softmax(dim=2)
        self.linear1 = nn.Linear(dim, dim)
        # self.linear2 = nn.Linear(256, dim)

    def forward(self, q, Layer=1):
        q = self.linear1(q)
        v = q
        k = q
        for i in range(Layer):

            u = torch.matmul(q, k.transpose(1, 2))
            u = u / self.scale

            attn = self.softmax(u)
            v = torch.matmul(v.transpose(1, 2), attn)
            v = v.transpose(1, 2)
            q = v
            k = v
        output = v
        return output

class labelattention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, dim=128):
        super().__init__()
        self.scale = math.sqrt(dim)
        self.softmax = nn.Softmax(dim=2)
        self.linearq1 = nn.Linear(dim, dim)
        self.linearq2 = nn.Linear(dim, dim)
        self.linearq3 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(768, 128)

    def forward(self, q1, q2, q3, k):
        q1 = self.linearq1(q1)
        v1 = q1
        q2 = self.linearq2(q2)
        v2 = q2
        q3 = self.linearq3(q3)
        v3 = q3

        k = self.linear2(k)


        u1 = torch.matmul(q1, k.transpose(0, 1))
        u1 = u1 / self.scale
        u2 = torch.matmul(q2, k.transpose(0, 1))
        u2 = u2 / self.scale
        u3 = torch.matmul(q3, k.transpose(0, 1))
        u3 = u3 / self.scale

        attn1 = self.softmax(u1)
        attn2 = self.softmax(u2)
        attn3 = self.softmax(u3)
        v1 = torch.matmul(v1.transpose(1, 2), attn1)
        v1 = v1.transpose(1, 2)
        v2 = torch.matmul(v2.transpose(1, 2), attn2)
        v2 = v2.transpose(1, 2)
        v3 = torch.matmul(v3.transpose(1, 2), attn3)
        v3 = v3.transpose(1, 2)
        v = v1 + v2 + v3
        return v.mean(1), u1.mean(1), u2.mean(1), u3.mean(1)

if __name__ == "__main__":
    n_q, n_k, n_v = 2, 4, 4
    d_q, d_k, d_v = 128, 128, 64
    batch = 32
    q = torch.randn(batch, n_q, d_q)
    k = torch.randn(batch, n_k, d_k)
    v = torch.randn(batch, n_v, d_v)
    mask = torch.zeros(batch, n_q, n_k).bool()

    attension = ScaledDotProductAttension(scale=np.power(d_k, 0.5))
    attn, output = attension(q, k, v, mask=mask)








