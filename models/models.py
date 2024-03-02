"""
@file: models.py
@author: wak
@date: 2024/2/18 11:53 
"""
from torch import nn
from torch.nn import functional as F

from configration.config import get_config


class NewModel(nn.Module):
    def __init__(self, vocab_size=21, *args, **kwargs):
        super(NewModel, self).__init__()
        cf = get_config()
        self.devicenum = cf.devicenum

        self.emb_dim = 512
        self.hidden_dim = 256
        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.lstm1 = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True,
                             dropout=0.5)

        self.conv = nn.Conv2d(1, 64, (32, self.emb_dim))

        self.dropout = nn.Dropout(0.5)

        # self.mlp = nn.Sequential(
        #     nn.Linear(64, 256),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 64),
        # )

        self.classification = nn.Sequential(
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.cuda(device=self.devicenum)
        x = self.embedding(x)  # [bs, 50, 512]
        x = self.transformer_encoder(x)
        x = self.transformer_encoder(x).permute(1, 0, 2)  # [50, bs, 512]
        x, _ = self.lstm1(x)  # [50, bs, 512]
        x = x.permute(1, 0, 2)  # [bs, 50, 512]
        x = x.view(x.size(0), 1, x.size(1), self.emb_dim)  # [bs,1, 50, 512]
        x = F.relu(self.conv(x))
        x = F.max_pool2d(input=x, kernel_size=(x.size(2), x.size(3)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        output = self.classification(x)
        return output
