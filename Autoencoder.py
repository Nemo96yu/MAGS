import torch
from torch import nn
import math
import torch.nn.functional as F
from Bi_Spe_Spa_Encoder import Encoder
from Spe_Spa_Decoder import Decoder
from attention import ScaledDotProductAttension, labelattention


class pooling(torch.nn.Module):
    def __init__(self):
        super(pooling, self).__init__()
        self.max = nn.MaxPool2d(2, stride=1)
        self.avg = nn.AvgPool2d(2, stride=1)

    def forward(self, x):
        return self.max(x) + self.avg(x)


class AttentionModule(nn.Module):

    def __init__(self, dim):
        super().__init__()
        # depth-wise convolution
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # depth-wise dilation convolution
        self.conv_spatial = nn.Conv2d(dim, dim, 3, stride=1, padding=2, groups=dim, dilation=2)
        # channel convolution (1×1 convolution)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        u = x.clone()
        attn0 = self.conv0(x)
        attn1 = self.conv_spatial(x)
        attn11 = self.conv1(x)
        attn = attn0 + attn1 + attn11
        attn = self.drop(attn)
        return u * attn


class AutoEncoder(torch.nn.Module):
    def __init__(self, spec_dim=200, in_chans=1, out_chans=128, in_dim=512, out_dim=256, num_class=16):
        super(AutoEncoder, self).__init__()

        # self.encoder_1d = nn.Sequential(
        #     nn.Linear(spec_dim, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 32),
        #     # nn.ReLU(True),
        #     # nn.Linear(12, 3)
        # )
        # self.decoder_1d = nn.Sequential(
        #     # nn.Linear(3, 12),
        #     # nn.ReLU(True),
        #     nn.Linear(32, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, spec_dim),
        #     nn.Tanh()
        # )
        #
        # self.label_embed = nn.Parameter(torch.randn((num_class, out_dim*2)))
        # self.encoder_2d = nn.Sequential(
        #     nn.Conv2d(3, out_chans, 3, stride=2, padding=1),  # b, 128, 16, 7, 7
        #     # nn.MaxPool2d(2, stride=1),  # b, 64, 6, 6
        #     pooling(),
        #     nn.ReLU(True),
        #     nn.Conv2d(out_chans, out_chans*2, 3, stride=1, padding=1),  # b, 256, 6, 6
        #     # nn.MaxPool2d(2, stride=1),  # b, 256, 5, 5
        #     pooling(),
        #     nn.ReLU(True),
        #     nn.Conv2d(out_chans * 2, out_chans * 2, 1)
        # )
        # self.decoder_2d = nn.Sequential(
        #     nn.ConvTranspose2d(out_chans * 2, out_chans, 3, stride=2),  # b, 16, 5, 5
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(out_chans, spec_dim, 3, stride=1),  # b, 8, 15, 15
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(spec_dim, 3, 1, stride=1),  # b, 1, 28, 28
        #     nn.Tanh()
        # )

        self.encoder = Encoder()
        self.decoder = Decoder()

        # self.attnention = AttentionModule(out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        # self.fc1 = nn.Linear(out_dim * 2, num_class)
        self.fc1 = nn.Sequential(
            nn.Linear(out_dim*2, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, num_class)
        )
        self.head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, out_dim*2)
        )
        self.labels = num_class
        # self.fc2 = nn.Linear(32, 32)
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, x, y, z,p, Flag):
        x = x.permute(0, 3, 1, 2)
        x_noisy = torch.empty_like(x)
        y_noisy = torch.empty_like(y)
        z = self.head(z)
        # if Flag:
        #     w = 1 - p/300 * 0.2
        # else:
        #     w = 0.8
        w = 0.7  # Pavia(0.7) Houston(0.9) Indian(0.7)
        for i in range(x.shape[-1]):
            noise = torch.randn([x.shape[0], x.shape[1], x.shape[2]], device=x.device) * 0.6     # Pavia(0.6) Houston(0.1) Indian(0.6)
            x_noisy[:, :, :, i] = x[:, :, :, i] + noise
        for i in range(y.shape[-1]):
            noise = torch.randn([y.shape[0]], device=x.device) * 0.6
            y_noisy[:, i] = y[:, i] + noise
        if Flag == True:
            # encoded1 = self.encoder_2d(x)
            # encoded2 = self.encoder_1d(y)
            encoded1, encoded2 = self.encoder(y, x)
            B, D, H, W = encoded2.size()
            # attn = self.norm2(self.attnention(encoded2).permute(0, 2, 3, 1)).reshape(B, -1, D)
            output1 = self.norm1(encoded2.permute(0, 2, 3, 1)).reshape(B, -1, D)
            output2 = self.norm2(encoded1)
            feature = torch.cat([output1.mean(1), output2.mean(1)], dim=-1)
            # score = torch.matmul(feature, z.transpose(0, 1))
            # _, loc = score.max(dim=1)
            # score = F.one_hot(loc.to(torch.int64), 16).float().cuda(8).detach()
            # feature = w*feature + (1-w)*torch.matmul(score, z)
            logistic = self.fc1(feature)
            return logistic
        else:
            # encoded1 = self.encoder_2d(x_noisy) + self.encoder_2d(x)
            # encoded2 = self.encoder_1d(y_noisy) + self.encoder_1d(y)
            encoded1, encoded2 = self.encoder(y, x)
            encoded11, encoded22 = self.encoder(y_noisy, x_noisy)

            B, D, H, W = encoded2.size()
            # attn = self.norm2(self.attnention(encoded2).permute(0, 2, 3, 1)).reshape(B, -1, D)
            output1 = self.norm1(encoded2.permute(0, 2, 3, 1)).reshape(B, -1, D)
            output2 = self.norm2(encoded1)
            output11 = self.norm1(encoded22.permute(0, 2, 3, 1)).reshape(B, -1, D)
            output22 = self.norm2(encoded11)
            feature1 = torch.cat([output1.mean(1), output2.mean(1)], dim=-1)
            feature2 = torch.cat([output11.mean(1), output22.mean(1)], dim=-1)


            score1 = F.normalize(torch.mm(feature1, z.t()), dim=-1)
            _, loc = score1.max(dim=1)
            m = F.one_hot(loc.to(torch.int64), self.labels).float().cuda(6)
            feature1 = w*feature1 + (1-w)*torch.matmul(m, z)

            score2 = F.normalize(torch.mm(feature2, z.t()), dim=-1)
            _, loc = score2.max(dim=1)
            m = F.one_hot(loc.to(torch.int64), self.labels).float().cuda(6)
            feature2 = w*feature2 + (1-w)*torch.matmul(m, z)

            logistic1 = self.fc1(feature1)
            logistic2 = self.fc1(feature2)
            # logistic = logistic1
            # decoded1 = self.decoder_2d(encoded1)
            # decoded2 = self.decoder_1d(encoded2)
            decoded1, decoded2 = self.decoder(encoded1, encoded2)
            return logistic1, decoded2.permute(0, 2, 3, 1), decoded1, logistic2, score1
