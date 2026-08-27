"""Segment-grouped batch sampler (spec 4.2): each batch = segments_per_batch segments x
frames_per_segment frames from each, so the herd-scale module sees within-segment consistency
pressure every step.  [M3]
"""
from __future__ import annotations

import random

import torch


class SegmentBatchSampler(torch.utils.data.Sampler):
    def __init__(self, by_segment: dict, segments_per_batch: int, frames_per_segment: int,
                 seed: int = 0, shuffle: bool = True):
        self.by_segment = {k: v for k, v in by_segment.items()
                           if len(v) >= frames_per_segment}
        self.spb = segments_per_batch
        self.fps = frames_per_segment
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __iter__(self):
        rng = random.Random(self.seed * 100003 + self.epoch)
        # each segment contributes floor(n / fps) groups per epoch
        groups = []
        for key, idxs in self.by_segment.items():
            idxs = list(idxs)
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs) - self.fps + 1, self.fps):
                groups.append(idxs[i:i + self.fps])
        if self.shuffle:
            rng.shuffle(groups)
        batch = []
        for g in groups:
            batch.extend(g)
            if len(batch) == self.spb * self.fps:
                yield batch
                batch = []

    def __len__(self):
        n_groups = sum(len(v) // self.fps for v in self.by_segment.values())
        return n_groups // self.spb
