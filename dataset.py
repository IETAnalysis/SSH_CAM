import torch
import numpy as np
import random
import json
import os
import logging
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Any


class StructuralMixupEngine:
    """
    Advanced engine for Sequence-Aware Structural Interpolation.
    Designed to maintain temporal causality while injecting spatial interference.
    """

    @staticmethod
    def mix(len_a: np.ndarray, time_a: np.ndarray, dir_a: np.ndarray,
            len_b: np.ndarray, time_b: np.ndarray, dir_b: np.ndarray,
            pkt_num: int, target_lam: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:

        # Guard: Ensure input integrity
        v_a = np.count_nonzero(len_a)
        v_b = np.count_nonzero(len_b)
        if v_a == 0 or v_b == 0:
            return len_a, time_a, dir_a, 1.0

        try:
            # Capacity-constrained partitioning: Eq. 11
            # max_m = min(available_b, capacity_limit, ratio_limit)
            limit_capacity = pkt_num * (1.0 - target_lam)
            limit_ratio = v_a * (1.0 - target_lam) / (target_lam + 1e-12)

            m_interference = int(np.floor(min(v_b, limit_capacity, limit_ratio)))

            # If interference is negligible, skip the grafting process
            if m_interference < 2:
                return len_a, time_a, dir_a, 1.0

            # Determine Target retention length k
            k_target = int(min(v_a, pkt_num - m_interference))
            k_target = max(1, min(k_target, pkt_num - 2))  # Structural safety bounds

            # Anchor Synthesis (Eq. 12): Graft B's space onto A's time trajectory
            l_anchor, d_anchor = len_b[0], dir_b[0]
            # Use target stream's next timestamp to mask the splice point
            t_anchor = time_a[min(k_target, len(time_a) - 1)]

            # Structural concatenation
            seg_l = np.concatenate((len_a[:k_target], [l_anchor], len_b[1:m_interference]))
            seg_d = np.concatenate((dir_a[:k_target], [d_anchor], dir_b[1:m_interference]))
            seg_t = np.concatenate((time_a[:k_target], [t_anchor], time_b[1:m_interference]))

            # Static memory allocation for output sequences
            out_l = np.zeros(pkt_num, dtype=np.int64)
            out_d = np.zeros(pkt_num, dtype=np.int64)
            out_t = np.zeros(pkt_num, dtype=np.float32)

            fill_limit = min(len(seg_l), pkt_num)
            out_l[:fill_limit], out_d[:fill_limit], out_t[:fill_limit] = seg_l[:fill_limit], seg_d[:fill_limit], seg_t[
                                                                                                                 :fill_limit]

            actual_lambda = k_target / (fill_limit + 1e-12)
            return out_l, out_t, out_d, actual_lambda

        except Exception as e:
            # Critical arithmetic failure recovery
            return len_a, time_a, dir_a, 1.0


class SSHCAMDataset(Dataset):
    def __init__(self, data_store: Dict[str, np.ndarray], pkt_num: int, lambda_val: float, mode: str):
        self.mode = mode
        self.pkt_num = pkt_num
        self.lambda_val = lambda_val

        # Verify schema requirements
        for key in ['l', 't', 'd', 'y']:
            if key not in data_store:
                raise KeyError(f"Schema violation: Mandatory field '{key}' missing from data_store.")

        self.l, self.t, self.d, self.y = data_store['l'], data_store['t'], data_store['d'], data_store['y']

        # Pre-calculate class indices for class-exclusion sampling
        unique_labels = np.unique(self.y)
        self.class_indices = {int(c): np.where(self.y == c)[0] for c in unique_labels}
        self.all_classes = list(self.class_indices.keys())

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        try:
            if self.mode == 'train' and self.lambda_val < 1.0:
                y_target = int(self.y[index])

                # Filter for heterogeneous interference (B != A)
                possible_classes = [c for c in self.all_classes if c != y_target]

                if not possible_classes:
                    # Fallback for degenerate single-class datasets
                    y_interference = y_target
                else:
                    y_interference = random.choice(possible_classes)

                # Online random sampling of interference stream B
                idx_interference = random.choice(self.class_indices[y_interference])

                ml, mt, md, lam = StructuralMixupEngine.mix(
                    self.l[index], self.t[index], self.d[index],
                    self.l[idx_interference], self.t[idx_interference], self.d[idx_interference],
                    self.pkt_num, self.lambda_val
                )
                return torch.LongTensor(ml), torch.FloatTensor(mt), torch.LongTensor(md), y_target, y_interference, lam

            # Baseline return for evaluation or clean training
            return torch.LongTensor(self.l[index]), torch.FloatTensor(self.t[index]), \
                torch.LongTensor(self.d[index]), int(self.y[index]), int(self.y[index]), 1.0

        except Exception as e:
            logging.error(f"Error at Dataset Index {index}: {str(e)}")
            # Return zero-padded fallback to prevent process crash
            zero_seq = np.zeros(self.pkt_num)
            return torch.LongTensor(zero_seq), torch.FloatTensor(zero_seq), torch.LongTensor(zero_seq), 0, 0, 1.0