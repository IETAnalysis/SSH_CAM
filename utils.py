import torch
import os
import logging
from datetime import datetime


class CurriculumScheduler:
    def __init__(self, l_sequence: list, min_epochs: int, threshold: float):
        self.l_sequence = l_sequence
        self.ptr = 0
        self.min_epochs = min_epochs
        self.threshold = threshold

        self.baseline_loss = None
        self.best_loss_stage = float('inf')
        self.stage_counter = 0

    def get_lambda(self) -> float:
        return self.l_sequence[self.ptr]

    def update_and_check(self, loss_val: float) -> bool:
        self.stage_counter += 1
        if self.baseline_loss is None:
            self.baseline_loss = loss_val

        if loss_val < self.best_loss_stage:
            self.best_loss_stage = loss_val

        # Evaluate Curriculum Gain G (Eq. 14)
        current_gain = self.baseline_loss - self.best_loss_stage

        if self.stage_counter >= self.min_epochs and current_gain <= self.threshold:
            if self.ptr < len(self.l_sequence) - 1:
                self.ptr += 1
                self._reset_stage()
                return True
        return False

    def _reset_stage(self):
        self.baseline_loss = None
        self.best_loss_stage = float('inf')
        self.stage_counter = 0


def setup_industrial_logger(log_dir: str):
    if log_dir is None:
        log_dir = f"./logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log = logging.getLogger("SSH_CAM_ENGINE")
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    fh = logging.FileHandler(os.path.join(log_dir, "runtime.log"))
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    log.addHandler(fh)
    log.addHandler(sh)
    return log