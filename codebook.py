import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from psk import PSK
from codebook_output import CodebookOutput

import torch_pruning as tp
from typing import Sequence

class Codebook(nn.Module):

    def __init__(self, latent_dim, n_embeddings=50, beta=1e-3, codebook_train_data=None, PSNR=10):

        super().__init__()

        self.latent_dim = latent_dim
        self.n_embeddings = n_embeddings
        self.mod = PSK(M=n_embeddings, PSNR=PSNR)
        self.beta = beta
        self.train_data = codebook_train_data
        # self.layer_name = layer_name

        self.embedding = nn.Parameter(torch.Tensor(n_embeddings, latent_dim))
        nn.init.uniform_(self.embedding, -1/ n_embeddings, 1 / n_embeddings)
    
    def compute_score(self, x):
        return x @ self.embedding.T / math.sqrt(self.latent_dim)
    
    def send_over_channel(self, x):
        return self.mod.demodulate(self.mod.awgn(self.mod.modulate(x)))
    
    def construct_noise(self, samples):
        x = samples.argmax(dim=-1)
        x_tilde = self.send_over_channel(x)
        noise = F.one_hot(x_tilde, num_classes=self.n_embeddings).float() -\
            F.one_hot(x, num_classes=self.n_embeddings).float()
        return noise
    
    def sample(self, score):
        dist = score.softmax(dim=-1)
        if self.training:
            samples = F.gumbel_softmax(score, tau=0.5, hard=True)
            noise = self.construct_noise(samples)
            samples = samples + noise
        else:
            samples = score.argmax(dim=-1)
            samples = self.send_over_channel(samples)
            samples = F.one_hot(samples, num_classes=self.n_embeddings).float()

        return samples, dist

    def forward(self, x):
        original_codebook_outputs = x
        was_codebook = type(x) == CodebookOutput

        if was_codebook:
            x = x.original_tensor

        initial_shape = x.shape
        if len(initial_shape) > 2:
            x_reshaped = x.view(-1, self.latent_dim)
        score = self.compute_score(x_reshaped)
        samples, dist = self.sample(score)

        res = samples @ self.embedding

        if len(initial_shape) > 2:
            res = res.view(*initial_shape)

        if was_codebook:
            codebook_outputs = original_codebook_outputs.codebook_outputs
            codebook_outputs.append([res, dist, self])
            output = CodebookOutput(original_codebook_outputs.original_tensor, codebook_outputs)
            return output
        else:
            output = CodebookOutput(x, [[res, dist, self]])
            return output

    def __repr__(self):
        return f'Codebook(latent_dim={self.latent_dim}, n_embeddings={self.n_embeddings})'
        
class CodebookPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer: Codebook, idxs: Sequence[int]) -> nn.Module:
        new_codebook = Codebook(layer.latent_dim, layer.n_embeddings - len(idxs), layer.beta, layer.train_data).to(next(layer.parameters()).device)
        new_codebook.embedding.data = layer.embedding.data[list(set(range(layer.n_embeddings)) - set(idxs))]
        return new_codebook
    
    def get_out_channels(self, layer):
        return layer.n_embeddings
    
    def prune_in_channels(self, layer: Codebook, idxs: Sequence[int]) -> nn.Module:
        new_codebook = Codebook(layer.latent_dim - len(idxs), layer.n_embeddings, layer.beta, layer.train_data).to(next(layer.parameters()).device)
        new_codebook.embedding.data = layer.embedding.data[:, list(set(range(layer.latent_dim)) - set(idxs))]
        return new_codebook

    def get_in_channels(self, layer):
        return layer.latent_dim