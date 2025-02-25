import torch
from torch import nn
import math


class Bi_attention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, dim=64):
        super().__init__()
        self.scale = math.sqrt(dim)
        self.softmax = nn.Softmax(dim=2)
        self.linearq1 = nn.Linear(dim, dim)
        self.linearq2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, y):
        B, D, H, W = y.size()
        y = y.permute(0, 2, 3, 1).reshape(B, -1, D)
        q1 = self.linearq1(x)
        q2 = self.linearq2(y)
        attn = torch.matmul(q1, q2.transpose(1, 2))
        attn = attn / self.scale
        score = self.drop(self.softmax(attn))
        v1 = torch.matmul(q1.transpose(1, 2), score)  # 200 *49
        v1 = v1.transpose(2, 1)  # 49 *200
        v2 = torch.matmul(q2.transpose(1, 2), score.transpose(1, 2))
        v2 = v2.transpose(1, 2)
        return v2, v1.reshape(B, H, W, D).permute(0, 3, 1, 2)


class pooling(torch.nn.Module):
    def __init__(self):
        super(pooling, self).__init__()
        self.max = nn.MaxPool2d(2, stride=1)
        self.avg = nn.AvgPool2d(2, stride=1)

    def forward(self, x):
        return self.max(x) + self.avg(x)


class Encoder(nn.Module):  # 64
    def __init__(self, in_chans=1, out_chans=64):
        super(Encoder, self).__init__()

        self.spe_embed = nn.Conv1d(in_channels=in_chans, out_channels=32, kernel_size=1)
        self.lstm_layer1 = nn.LSTM(input_size=32, hidden_size=32, num_layers=1,
                                   bias=True, batch_first=True, dropout=0, bidirectional=True, proj_size=0)
        self.act1 = nn.ReLU()
        self.lstm_layer2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1,
                                   bias=True, batch_first=True, dropout=0, bidirectional=True, proj_size=0)
        self.act2 = nn.ReLU()
        self.lstm_layer3 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1,
                                   bias=True, batch_first=True, dropout=0, bidirectional=True, proj_size=0)
        self.act3 = nn.ReLU()
        # b, 128, 16, 7, 7
        self.conv_layer1 = nn.Sequential(nn.Conv2d(32, out_chans, 3, stride=2, padding=1), pooling(), nn.ReLU(True))
        # nn.MaxPool2d(2, stride=1),  # b, 256, 5, 5
        self.conv_layer2 = nn.Sequential(nn.Conv2d(out_chans, out_chans * 2, 3, stride=1, padding=1), pooling(),
                                         nn.ReLU(True))
        self.conv_layer3 = nn.Conv2d(out_chans * 2, out_chans * 4, 1)
        self.attention_layer1 = Bi_attention(dim=64)
        self.attention_layer2 = Bi_attention(dim=128)
        # self.attention_layer3 = Bi_attention(dim=256)
        self.init_weight()

    def init_weight(self):
        for name, param in self.lstm_layer1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)
        for name, param in self.lstm_layer2.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)
        for name, param in self.lstm_layer3.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)

    def forward(self, img_spe, img_spa):
        img_spe = img_spe.view(img_spe.shape[0], 1, img_spe.shape[1])
        # img_spa = img_spa.permute(0, 3, 1, 2)
        spe_embedding = self.spe_embed(img_spe)  # B, 200,  32
        spe_embedding = spe_embedding.permute(0, 2, 1)  # B, 32, 200
        # Layer 1
        spe1, _ = self.lstm_layer1(spe_embedding)
        spe1 = self.act1(spe1)
        spa1 = self.conv_layer1(img_spa)
        attn_spe, attn_spa = self.attention_layer1(spe1, spa1)
        spe1 = spe1 + 0.3*attn_spe
        spa1 = spa1 + 0.3*attn_spa
        # Layer 2
        spe2, _ = self.lstm_layer2(spe1)
        spe2 = self.act2(spe2)
        spa2 = self.conv_layer2(spa1)
        attn_spe, attn_spa = self.attention_layer2(spe2, spa2)
        spe2 = spe2 + 0.3*attn_spe
        spa2 = spa2 + 0.3*attn_spa
        # Layer 3
        spe3, _ = self.lstm_layer3(spe2)
        spe3 = self.act3(spe3)
        spa3 = self.conv_layer3(spa2)
        # attn_spe, attn_spa = self.attention_layer3(spe3, spa3)
        # spe3 = spe3 + attn_spe
        # spa3 = spa3 + attn_spa
        return spe3, spa3
