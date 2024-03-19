"""
@file: models.py
@author: wak
@date: 2024/2/18 11:53 
"""
import numpy as np
import torch
from mamba_ssm import Mamba
# from mamba_ssm import Mamba
from torch import nn
from torch.nn import functional as F

from configration.config import get_config

device = get_config().device


class BaseModel(nn.Module):
    def __init__(self, vocab_size=21):
        super(BaseModel, self).__init__()
        cf = get_config()
        self.device = torch.device("cuda", cf.devicenum)
        self.emb_dim = 512
        self.hidden_dim = 25
        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        # self.mamba = Mamba(
        #     d_model=self.emb_dim,
        #     d_state=16,
        #     d_conv=4, expand=2
        # ).to(self.device)
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2,
                          bidirectional=True, dropout=0.4)


class NewModel(nn.Module):
    def __init__(self, vocab_size=21):
        super(NewModel, self).__init__()
        cf = get_config()
        self.device = torch.device('cuda', cf.devicenum)
        # self.filter_sizes = [1, 2, 3, 4, 6, 8, 16, 32]  # Contrast
        self.emb_dim = 100
        self.hidden_dim = 50
        self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)

        # self.mamba = Mamba(
        #     d_model=self.emb_dim,
        #     d_state=16,
        #     d_conv=4, expand=2
        # ).to(self.device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.lstm1 = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True,
                             dropout=0.5)

        self.conv = nn.Conv2d(1, 64, (32, self.emb_dim))

        self.dropout = nn.Dropout(0.5)

        # self.mlp = nn.Sequential(
        #     nn.Linear(5000, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.LeakyReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(),
        #     nn.Linear(1024, 64)
        # )

        self.classification = nn.Sequential(
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.embedding(x)  # [bs, 50, 100]
        x = self.transformer_encoder(x)
        x = self.transformer_encoder(x).permute(1, 0, 2)  # [50, bs, 100]
        # x = self.mamba(x).permute(1, 0, 2)  # [50, bs, 100]
        x, _ = self.lstm1(x)  # [50, bs, 100]
        x = x.permute(1, 0, 2)  # [bs, 50, 100]
        x = x.view(x.size(0), 1, x.size(1), self.emb_dim)  # [bs,1, 50, 100]
        x = F.relu(self.conv(x))
        x = F.max_pool2d(input=x, kernel_size=(x.size(2), x.size(3)))
        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        return x

    def trainModel(self, x):
        output = self.forward(x)
        return self.classification(output)


# [0.91927, 0.96, 0.875, 0.96354, 0.91553, 0.96498, 0.84185]
# [0.92188, 0.94475, 0.89529, 0.94819, 0.91935, 0.96067, 0.84485]
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        cf = get_config()
        self.device = torch.device('cuda', cf.devicenum)
        # self.mamba = Mamba(
        #     d_model=531,
        #     d_state=16,
        #     d_conv=4, expand=2
        # ).to(self.device)
        self.convs = nn.Sequential(
            nn.Conv1d(531, 64, kernel_size=3, stride=1, padding=1),  # [bs, 64, 50]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [bs, 64, 25]
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),  # [bs, 128, 25]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # [bs, 128, 12]
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=32, num_layers=1, bidirectional=True,
                            dropout=0.5)

        self.block1 = nn.Sequential(
            nn.Linear(12 * 64, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64)
        )
        self.classification = nn.Sequential(
            nn.Linear(64, 2),
        )

    def forward(self, aa):
        # x [bs, 50, 531]
        aa = aa.permute(0, 2, 1)  # [bs, 531, 50]
        x = self.convs(aa)
        x = x.permute(0, 2, 1)  # [bs, 12, 128]
        output, _ = self.lstm(x)
        output = output.view(output.size(0), -1)
        output = self.block1(output)
        return output

    def trainModel(self, x):
        with torch.no_grad():
            x = self.forward(x)

        return self.classification(x)


class MyModel4(nn.Module):
    def __init__(self, vocab_size=21):
        super(MyModel4, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(531, 64, kernel_size=3, stride=1, padding=1),  # [bs, 64, 50]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [bs, 64, 25]
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),  # [bs, 128, 25]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [bs, 128, 12]
        )
        self.fc1 = nn.Linear(12 * 128, 1024)
        self.emb_dim = 512
        self.hidden_dim = 25
        self.emb = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2,
                          bidirectional=True, dropout=0.4)
        self.block1 = nn.Sequential(nn.Linear(4036, 2048),
                                    nn.BatchNorm1d(2048),
                                    nn.LeakyReLU(),
                                    nn.Linear(2048, 1024),
                                    )
        self.classification = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 640),
            nn.BatchNorm1d(640),
            nn.LeakyReLU(),
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, data):
        oe, aa = data['oe'], data['aa']
        # [bs, 50, 531] 和 [bs, 50, 531] concat 再通过 1-D 卷积
        aa = aa.permute(0, 2, 1)
        aa_output = self.convs(aa)  # [bs, 128, 12]
        aa_output = aa_output.view(aa_output.size(0), -1)  # [bs, 1536]

        oe_output = self.emb(oe)  # [bs, 50, 512]
        output = self.transformer_encoder(oe_output).permute(1, 0, 2)  # [50, bs, 512]
        output, _ = self.gru(output)  # [50, bs, 50]
        output = output.permute(1, 0, 2)  # [bs, 50, 50]
        output = output.reshape(output.shape[0], -1)  # [bs, 2500]
        output = torch.cat((aa_output, output), dim=1)

        return self.block1(output)

    def trainModel(self, data):
        with torch.no_grad():
            out = self.forward(data)
        output = torch.cat((data['bert'], out), dim=1)
        return self.classification(output)


class MyModel5(nn.Module):
    def __init__(self, vocab_size=21):
        super(MyModel5, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(531, 64, kernel_size=3, stride=1, padding=1),  # [bs, 64, 50]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [bs, 64, 25]
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),  # [bs, 128, 25]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [bs, 128, 12]
        )
        self.mamba = Mamba(
            d_model=512,
            d_state=16,
            d_conv=2,
            expand=2
        ).to(device)
        self.emb_dim = 512
        self.hidden_dim = 25
        self.emb = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2,
                          bidirectional=True, dropout=0.4)
        self.block1 = nn.Sequential(nn.Linear(4036, 2048),
                                    nn.BatchNorm1d(2048),
                                    nn.LeakyReLU(),
                                    nn.Linear(2048, 1024),
                                    )
        self.classification = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 640),
            nn.BatchNorm1d(640),
            nn.LeakyReLU(),
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, data):
        oe, aa = data['oe'], data['aa']
        # [bs, 50, 531] 和 [bs, 50, 531] concat 再通过 1-D 卷积
        aa = aa.permute(0, 2, 1)
        aa_output = self.convs(aa)  # [bs, 128, 12]
        aa_output = aa_output.view(aa_output.size(0), -1)  # [bs, 1536]

        oe_output = self.emb(oe)  # [bs, 50, 512]
        output = self.mamba(oe_output).permute(1, 0, 2)  # [50, bs, 512]
        output, _ = self.gru(output)  # [50, bs, 50]
        output = output.permute(1, 0, 2)  # [bs, 50, 50]
        output = output.reshape(output.shape[0], -1)  # [bs, 2500]
        output = torch.cat((aa_output, output), dim=1)

        return self.block1(output)

    def trainModel(self, data):
        with torch.no_grad():
            output = self.forward(data)
        output = torch.cat((data['bert'], output), dim=1)
        return self.classification(output)


class BFD(nn.Module):
    def __init__(self):
        super(BFD, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(1024, 640),
            nn.BatchNorm1d(640),
            nn.LeakyReLU(),
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.block(x)

    def trainModel(self, x):
        return self.forward(x)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x


class MyModel2(nn.Module):
    def __init__(self):
        super(MyModel2, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64)
        )

        self.classification = nn.Sequential(
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.block1(x)

    def trainModel(self, x):
        with torch.no_grad():
            output = self.forward(x)
        return self.classification(output)


class MyModel3(nn.Module):
    def __init__(self):
        super(MyModel3, self).__init__()
        self.att1 = CoAttention(20)
        self.att2 = CoAttention(531)
        self.block1 = nn.Sequential(
            nn.Linear(1575, 1024),
            nn.ReLU(),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(512, 128),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(),
            # nn.Linear(256, 64)
        )
        self.classification = nn.Linear(128, 2)

    def forward(self, bert, aaindex, blosum):
        fusion1 = self.att1(bert, blosum)
        fusion2 = self.att2(bert, aaindex)
        f = torch.cat((bert, fusion1, fusion2), dim=1)
        return self.block1(f)

    def trainModel(self, bert, aaindex, blosum):
        with torch.no_grad():
            output = self.forward(bert, aaindex, blosum)
        return self.classification(output)


class CoAttention(nn.Module):
    def __init__(self, dim):
        super(CoAttention, self).__init__()
        self.Q = nn.Linear(1024, dim)
        self.K = nn.Linear(dim, dim)

    def forward(self, data1, data2):
        query = self.Q(data1)  # [batch_size, dim]
        key = self.K(data2)  # [batch_size, dim]
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(1024)  # [batch_size, batch_size]
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, batch_size]
        context = torch.matmul(attn, key) + query  # [batch_size, dim]
        return context


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class MyModel6(nn.Module):
    def __init__(self, vocab_size=21):
        super(MyModel6, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(581, 64, kernel_size=3, stride=1, padding=1),  # [bs, 64, 50]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [bs, 64, 25]
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),  # [bs, 128, 25]
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [bs, 128, 12]
        )
        self.mamba = Mamba(
            d_model=512,
            d_state=16,
            d_conv=2, expand=2
        ).to(device)
        self.emb_dim = 512
        self.hidden_dim = 25
        self.emb = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2,
                          bidirectional=True, dropout=0.4)
        self.block1 = nn.Sequential(
            nn.Linear(1536, 1024),
        )
        self.classification = nn.Sequential(
            # nn.Linear(2048, 1024),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(),
            nn.Linear(1024, 640),
            nn.BatchNorm1d(640),
            nn.LeakyReLU(),
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, data):
        oe, aa = data['oe'], data['aa']
        # [bs, 50, 531] 和 [bs, 50, 531] concat 再通过 1-D 卷积
        # aa = aa.permute(0, 2, 1)
        # aa_output = self.convs(aa)  # [bs, 128, 12]
        # aa_output = aa_output.view(aa_output.size(0), -1)  # [bs, 1536]

        oe_output = self.emb(oe)  # [bs, 50, 512]
        output = self.transformer_encoder(oe_output).permute(1, 0, 2)  # [50, bs, 512]
        output, _ = self.gru(output)  # [50, bs, 50]
        output = output.permute(1, 0, 2)  # [bs, 50, 50]
        output = torch.cat([output, aa], dim=2).permute(0, 2, 1)  # [bs, 581, 50]
        aa_output = self.convs(output)  # [bs, 128, 12]
        aa_output = aa_output.reshape(output.shape[0], -1)  # [bs, 2500]

        return self.block1(aa_output)

    def trainModel(self, data):
        with torch.no_grad():
            output = self.forward(data)
        # output = torch.cat((data['bert'], output), dim=1)
        return self.classification(output)


class MambaTest(torch.nn.Module):
    def __init__(self, vocab_size=21, d_state=21, d_conv=2, expand=8):
        super(MambaTest, self).__init__()
        self.emb_dim = 512
        self.hidden_dim = 25
        self.emb = nn.Embedding(vocab_size, self.emb_dim, padding_idx=0)
        self.encoder = Mamba(
            d_model=512,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=True
        ).to(device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.encoder1 = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, num_layers=2,
                          bidirectional=True, dropout=0.4)
        self.classification = nn.Sequential(
            nn.Linear(2500, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 640),
            nn.BatchNorm1d(640),
            nn.LeakyReLU(),
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, data):
        oe = data['oe']
        emb = self.emb(oe)
        encoding = self.encoder(emb)
        output, _ = self.gru(encoding)
        return output.reshape(output.shape[0], -1)

    def trainModel(self, data):
        return self.classification(self.forward(data))
