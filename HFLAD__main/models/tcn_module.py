import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class HierarchicalTimeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dropout=0.3):
        super(HierarchicalTimeEncoder, self).__init__()

        self.causal_conv = nn.Sequential(
            weight_norm(nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size - 1)),
            Chomp1d(kernel_size - 1),
            nn.ReLU()
        )
        dilation_b = 4
        padding_b = (kernel_size - 1) * dilation_b
        self.dilated_conv = nn.Sequential(
            weight_norm(nn.Conv1d(input_dim, hidden_dim, kernel_size,
                                  padding=padding_b, dilation=dilation_b)),
            Chomp1d(padding_b),
            nn.ReLU()
        )
        self.tcn_block = TemporalBlock(input_dim, hidden_dim, kernel_size,
                                       stride=1, dilation=8, padding=(kernel_size - 1) * 8,
                                       dropout=dropout)

    def forward(self, x):
        a = self.causal_conv(x)
        b = self.dilated_conv(x)
        c = self.tcn_block(x)

        return a, b, c
