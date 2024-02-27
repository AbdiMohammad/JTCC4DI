import torch
from dataclasses import dataclass

@dataclass
class CodebookOutput:
    original_tensor: torch.Tensor
    codebook_outputs: list

    def map(self, f):
        out_original = f(self.original_tensor)
        codebook_outputs = []
        for codebook_x, dist, codebook in self.codebook_outputs:
            codebook_outputs.append([f(codebook_x), dist, codebook])
        return CodebookOutput(out_original, codebook_outputs)
    
    def grad_fn(self):
        return self.original_tensor.grad_fn
    
    def __iter__(self):
        self.iterable = [self.original_tensor]
        return iter(self.iterable)

    def __next__(self):
        return next(self.iterable)