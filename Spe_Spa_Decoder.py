import torch
from torch import nn


class Decoder(torch.nn.Module):
    def __init__(self, spec_dim=32, out_chans=128):
        super(Decoder, self).__init__()
        self.layer1 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1,
                              bias=True, batch_first=True, dropout=0, bidirectional=False, proj_size=0)
        self.act1 = nn.ReLU()
        self.layer2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1,
                              bias=True, batch_first=True, dropout=0, bidirectional=False, proj_size=0)
        self.act2 = nn.ReLU()
        self.layer3 = nn.LSTM(input_size=64, hidden_size=1, num_layers=1,
                              bias=True, batch_first=True, dropout=0, bidirectional=False, proj_size=0)
        self.act3 = nn.Tanh()

        self.decoder_spa = nn.Sequential(
            nn.ConvTranspose2d(out_chans * 2, out_chans, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(out_chans, spec_dim, 3, stride=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(spec_dim, spec_dim, 1, stride=1),  # b, 1, 28, 28
            nn.Tanh()
        )
        # self.norm1 = nn.LayerNorm(out_dim)
        # self.norm2 = nn.LayerNorm(out_dim)

        self.init_weight()

    def init_weight(self):
        for name, param in self.layer1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)

        for name, param in self.layer2.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)
        for name, param in self.layer3.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)


    def forward(self, x, y):
        decoded_spe1, _ = self.layer1(x)
        decoded_spe1 = self.act1(decoded_spe1)
        decoded_spe2, _ = self.layer2(decoded_spe1)
        decoded_spe2 = self.act2(decoded_spe2)
        decoded_spe3, _ = self.layer3(decoded_spe2)
        decoded_spe = self.act3(decoded_spe3)
        decoded_spa = self.decoder_spa(y)
        return decoded_spe, decoded_spa
