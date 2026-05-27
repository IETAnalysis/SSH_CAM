import torch
import numpy as np
import random
from torch.utils.data import Dataset
from typing import Dict, Tuple


class StructuralMixupEngine:
    @staticmethod
    def mix(len_a: np.ndarray, time_a: np.ndarray, dir_a: np.ndarray,
            len_b: np.ndarray, time_b: np.ndarray, dir_b: np.ndarray,
            pkt_num: int, lambda_val: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        active_a = np.count_nonzero(len_a)
        active_b = np.count_nonzero(len_b)

        # Interference injection length m
        m = int(min(active_b, pkt_num * (1 - lambda_val)))
        if m < 2:
            return len_a, time_a, dir_a, 1.0

        # Target stream preservation length k
        k = min(active_a, pkt_num - m)

        # Synthesize Anchor Packet (Spatial: B, Temporal: A)
        l_anchor, d_anchor = len_b[0], dir_b[0]
        t_anchor = time_a[min(k, pkt_num - 1)]

        # Sequence concatenation
        mixed_l = np.concatenate((len_a[:k], [l_anchor], len_b[1:m]))
        mixed_d = np.concatenate((dir_a[:k], [d_anchor], dir_b[1:m]))
        mixed_t = np.concatenate((time_a[:k], [t_anchor], time_b[1:m]))

        # Zero-padding alignment
        o_l, o_d, o_t = np.zeros(pkt_num), np.zeros(pkt_num), np.zeros(pkt_num)
        limit = min(len(mixed_l), pkt_num)
        o_l[:limit], o_d[:limit], o_t[:limit] = mixed_l[:limit], mixed_d[:limit], mixed_t[:limit]

        return o_l, o_t, o_d, k / limit


class TrainDataset(Dataset):

    def __init__(self, data_dict: Dict[str, np.ndarray], pkt_num: int, lambda_val: float):
        self.pkt_num = pkt_num
        self.lambda_val = lambda_val

        self.lengths = data_dict['length']
        self.times = data_dict['time']
        self.dirs = data_dict['dir']
        self.labels = data_dict['label']

        # Build class indices for fast cross-class sampling
        self.class_indices = {int(c): np.where(self.labels == c)[0] for c in np.unique(self.labels)}
        self.all_classes = list(self.class_indices.keys())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        y_a = int(self.labels[idx])

        # Randomly select a different class for interference
        other_classes = [c for c in self.all_classes if c != y_a]
        y_b = random.choice(other_classes) if other_classes else y_a
        idx_b = random.choice(self.class_indices[y_b])

        # Execute dynamic structural mixup
        ml, mt, md, act_lam = StructuralMixupEngine.mix(
            self.lengths[idx], self.times[idx], self.dirs[idx],
            self.lengths[idx_b], self.times[idx_b], self.dirs[idx_b],
            self.pkt_num, self.lambda_val
        )
        return torch.LongTensor(ml), torch.FloatTensor(mt), torch.LongTensor(md), y_a, y_b, act_lam


class EvaluationDataset(Dataset):

    def __init__(self, pre_mixed_dict: Dict[str, np.ndarray]):
        self.lengths = pre_mixed_dict['length']
        self.times = pre_mixed_dict['time']
        self.dirs = pre_mixed_dict['dir']
        self.label_a = pre_mixed_dict['label_a']
        self.label_b = pre_mixed_dict['label_b']
        self.lam = pre_mixed_dict['lam']

    def __len__(self):
        return len(self.label_a)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.lengths[idx]),
            torch.FloatTensor(self.times[idx]),
            torch.LongTensor(self.dirs[idx]),
            int(self.label_a[idx]),
            int(self.label_b[idx]),
            float(self.lam[idx])
        )


