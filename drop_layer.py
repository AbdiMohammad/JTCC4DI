import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp
from typing import Sequence

class DropLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels
        self.dummy_weights = torch.eye(self.in_channels).unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        self.dummy_weights = self.dummy_weights.to(x.device)
        return F.conv2d(x, self.dummy_weights)
    
    def __repr__(self):
        return f"DropLayer(in_channels={self.in_channels}, out_channels={self.out_channels})"

class DropLayerPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer: DropLayer, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
        keep_idxs.sort()
        layer.out_channels -= len(idxs)
        layer.dummy_weights = torch.index_select(layer.dummy_weights, 0, torch.LongTensor(keep_idxs).to(layer.dummy_weights.device)).to(layer.dummy_weights.device)
        return layer
    
    def get_out_channels(self, layer):
        return layer.out_channels
    
    def prune_in_channels(self, layer: DropLayer, idxs: Sequence[int]) -> nn.Module:
        keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
        keep_idxs.sort()
        layer.in_channels -= len(idxs)
        layer.dummy_weights = torch.index_select(layer.dummy_weights, 1, torch.LongTensor(keep_idxs).to(layer.dummy_weights.device)).to(layer.dummy_weights.device)
        return layer

    def get_in_channels(self, layer):
        return layer.in_channels