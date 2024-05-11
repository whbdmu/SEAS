# -*- coding: UTF-8 -*-

import torch
from torch import Tensor
from torch.nn import (
    Module, init, Sequential, Linear, BatchNorm1d, Identity, Flatten
)
from torchvision.ops.misc import Permute
from typing import List, Optional
from collections import OrderedDict


class CrossAttention(Module):
    def __init__(self, pool: Module, embedder: Optional[Module], norm: Optional[Module], decoder: Module):
        super().__init__()
        self.pool = pool
        self.flatten = Flatten(start_dim=2, end_dim=-1)
        self.permute = Permute([0, 2, 1])
        self.embedder = embedder or Identity()
        self.norm = norm or Identity()
        self.decoder = decoder

    def reset_parameters(self) -> None:
        getattr(self.embedder, 'reset_parameters', lambda: None)()
        getattr(self.norm, 'reset_parameters', lambda: None)()
        self.decoder.reset_parameters()

    def forward(self, embeddings: Tensor, noise_maps: Tensor, num_embs_per_map: List[int]) -> Tensor:
        noise_maps = self.pool(noise_maps)
        noise_maps = self.flatten(noise_maps)
        noise_maps = self.permute(noise_maps)
        noise_maps = self.embedder(noise_maps)
        noise_maps = self.norm(noise_maps)
        return embeddings + self.decoder(embeddings.detach(), noise_maps, num_embs_per_map)


class LinearProjection(Module):
    def __init__(self, dim: int, memery_channels: int):
        super().__init__()
        self.projector = Sequential(OrderedDict([
            ('fc', Linear(memery_channels, dim, bias=True)),
            ('bn', BatchNorm1d(dim)),
        ]))

    def reset_parameters(self) -> None:
        init.normal_(self.projector.fc.weight, std=0.01)
        init.zeros_(self.projector.fc.bias)
        self.projector.bn.reset_parameters()

    def forward(self, embeddings: Tensor, noise_maps: Tensor, num_embs_per_map: List[int]) -> Tensor:
        noise_vectors = torch.amax(noise_maps, dim=(2, 3))
        noise_vectors = torch.repeat_interleave(
            noise_vectors,
            torch.tensor(num_embs_per_map, device=noise_vectors.device),
            dim=0,
        )
        return embeddings + self.projector(noise_vectors)
