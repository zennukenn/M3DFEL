import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange
from utils import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TemporalAttention(nn.Module):
    def __init__(self, d_model, seq_len):
        super(TemporalAttention, self).__init__()
        self.fc1 = nn.Linear(d_model * seq_len, d_model)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model, seq_len)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        # Concatenate along the feature dimension
        x_concat = x.reshape(batch_size, -1)
        # Fully connected layer and ReLU activation
        x_fc = self.fc1(x_concat)
        x_relu = self.relu(x_fc)
        # Learn weights
        weights = self.fc2(x_relu)
        weights = F.softmax(weights, dim=1)  # Apply softmax to normalize weights
        weights = weights.view(batch_size, seq_len, 1)
        # Apply weights to the original transformer output
        weighted_output = x * weights
        return weighted_output


class TemporalEmbeddingSubnetwork(nn.Module):
    def __init__(self, d_model=512, nhead=8, seq_len=7, num_layers=3, dropout=0.1):
        super(TemporalEmbeddingSubnetwork, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=seq_len)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.temporal_attention = TemporalAttention(d_model, seq_len)
        self.fc = nn.Linear(d_model, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x.permute(1, 0, 2)  # Change shape to [seq_len, batch_size, d_model]
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Change back to [batch_size, seq_len, d_model]
        x = self.temporal_attention(x)
        x = x.permute(0, 2, 1)  # Permute to [batch_size, d_model, seq_len]
        x = self.gap(x).squeeze(-1)  # Global Average Pooling
        output = self.fc(x)
        return output


class M3DFEL(nn.Module):
    """The proposed M3DFEL framework

    Args:
        args
    """

    def __init__(self, args):
        super(M3DFEL, self).__init__()

        self.args = args
        self.device = torch.device(
            'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.bag_size = self.args.num_frames // self.args.instance_length
        self.instance_length = self.args.instance_length

        # backbone networks
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.features = nn.Sequential(
            *list(model.children())[:-1])  # after avgpool 512x1

        # Temporal Embedding Subnetwork
        d_model = 512
        nhead = 8
        num_layers = 3
        seq_len = self.bag_size
        self.temporal_embedding_subnetwork = TemporalEmbeddingSubnetwork(
            d_model=d_model, nhead=nhead, seq_len=seq_len, num_layers=num_layers
        )

        self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)

        # classifier
        self.fc = nn.Linear(1024, self.args.num_classes)
        self.Softmax = nn.Softmax(dim=-1)

    def MIL(self, x):
        """The Multi Instance Learning Agregation of instances

        Inputs:
            x: [batch, bag_size, 512]
        """
        # Use Temporal Embedding Subnetwork
        x = self.temporal_embedding_subnetwork(x)
        return x

    def forward(self, x):
        # [batch, 16, 3, 112, 112]
        x = rearrange(x, 'b (t1 t2) c h w -> (b t1) c t2 h w',
                      t1=self.bag_size, t2=self.instance_length)
        # [batch*bag_size, 3, il, 112, 112]

        x = self.features(x).squeeze()
        # [batch*bag_size, 512]
        x = rearrange(x, '(b t) c -> b t c', t=self.bag_size)

        # [batch, bag_size, 512]
        out = self.MIL(x)
        # [batch, bag_size, 1024]

        #         x = self.pwconv(x).squeeze()
        #         # [batch, 1024]

        #         # [batch, num_classes, 4]
        #         out = self.fc(x)
        #         # [batch, 7]

        return out

