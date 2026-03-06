import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import os
import sys
from model import SSHCAMSystem
from dataset import SSHCAMDataset
from utils import CurriculumScheduler, setup_industrial_logger


class SSHCAMTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        self.logger = setup_industrial_logger(args.save_path)
        self.writer = SummaryWriter(log_dir=args.save_path)

        self.data_store = self._init_data_store(args.input_json)
        self.model = SSHCAMSystem(args).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-5)

        # Build Curriculum Schedule (1.0 -> target_lambda)
        schedule = [round(x, 2) for x in np.arange(1.0, args.min_lambda - 0.01, -0.05).tolist()]
        self.scheduler = CurriculumScheduler(schedule, args.min_stage_epochs, args.epsilon)

    def _init_data_store(self, path):
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Input JSON path '{path}' is invalid.")

        with open(path, 'r') as f:
            raw = json.load(f)

        return {
            'l': np.array([i['length'] for i in raw]),
            't': np.array([i['time'] for i in raw]),
            'd': np.array([i['dir'] for i in raw]),
            'y': np.array([i['label'] for i in raw])
        }

    def compute_loss_manifold(self, output, y_a, y_b, lam, beta):
        if self.args.loss_type == 'gmm':
            z_feat = output
            # Matrix-form Euclidean Distance Optimization
            z_sq = torch.sum(z_feat ** 2, dim=1, keepdim=True)
            mu_sq = torch.sum(self.model.gmm_centroids ** 2, dim=1, keepdim=True).T
            dist_sq_matrix = torch.clamp(z_sq - 2 * torch.matmul(z_feat, self.model.gmm_centroids.T) + mu_sq, min=0.0)

            def compute_dist_ce(targets):
                neg_logits = -0.5 * dist_sq_matrix
                margin_adj = torch.zeros_like(neg_logits).scatter_(1, targets.unsqueeze(1), self.args.alpha)
                return F.cross_entropy(neg_logits * (1.0 + margin_adj), targets, reduction='none')

            loss_dis = torch.mean(lam * compute_dist_ce(y_a) + (1.0 - lam) * compute_dist_ce(y_b))

            # Geometric Anchor Regularization (Eq. 19)
            dist_a = torch.gather(dist_sq_matrix, 1, y_a.unsqueeze(1)).squeeze()
            dist_b = torch.gather(dist_sq_matrix, 1, y_b.unsqueeze(1)).squeeze()
            loss_reg = 0.5 * torch.mean(lam * dist_a + (1.0 - lam) * dist_b)

            return loss_dis + beta * loss_reg, loss_dis, loss_reg
        else:
            logits, _ = output
            loss = torch.mean(lam * F.cross_entropy(logits, y_a) + (1.0 - lam) * F.cross_entropy(logits, y_b))
            return loss, loss, torch.tensor(0.0)

    def train(self):
        total_steps = 0
        for ep in range(self.args.epochs):
            cur_lam = self.scheduler.get_lambda()
            ds_train = SSHCAMDataset(self.data_store, self.args.pkt_num, cur_lam, 'train')
            loader = DataLoader(ds_train, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers)

            self.model.train()
            cur_beta = self.args.eta * self.scheduler.stage_counter  # Eq. 24

            epoch_loss_accum = []
            pbar = tqdm(loader, desc=f"Ep {ep} | λ={cur_lam}")

            for l, t, d, ya, yb, batch_lam in pbar:
                l, t, d, ya, yb, batch_lam = [x.to(self.device) for x in [l, t, d, ya, yb, batch_lam]]
                mask = (l == 0) if self.args.backbone == 'transformer' else None

                try:
                    out = self.model(l, t, d, mask)
                    loss, ld, lr = self.compute_loss_manifold(out, ya, yb, batch_lam, cur_beta)

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    epoch_loss_accum.append(loss.item())
                    if total_steps % 50 == 0:
                        self.writer.add_scalar("Train/Loss_Total", loss.item(), total_steps)
                        self.writer.add_scalar("Train/Cur_Lambda", cur_lam, total_steps)
                    total_steps += 1
                    pbar.set_postfix(Loss=f"{loss.item():.4f}")
                except Exception as e:
                    self.logger.error(f"Batch execution error at Step {total_steps}: {e}")
                    continue

            # Curriculum Logic
            avg_ep_loss = np.mean(epoch_loss_accum) if epoch_loss_accum else 0.0
            if self.scheduler.update_and_check(avg_ep_loss):
                self.logger.info(f"[Curriculum Advance] Switching to Lambda = {self.scheduler.get_lambda()}")
                torch.save(self.model.state_dict(), os.path.join(self.args.save_path, f"ckpt_stage_lam_{cur_lam}.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SSH-CAM Professional CLI Interface")

    # Critical Path Configurations (No Defaults)
    p.add_argument("--input_json", type=str, required=True, help="Path to the JSON dataset.")
    p.add_argument("--save_path", type=str, default=None, help="Directory to save logs and ckpts.")

    # Architecture and Backend Logic
    p.add_argument("--backbone", type=str, choices=["transformer", "gru"], default=None)
    p.add_argument("--loss_type", type=str, choices=["gmm", "softmax"], default=None)
    p.add_argument("--time_enc", type=str, choices=["cam", "log"], default=None)

    # Model Hyper-Parameters (Placeholders)
    p.add_argument("--vocab_size", type=int, default=None)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--nhead", type=int, default=None)  # Keep for Transformer stability
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)

    # Execution Environment
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pkt_num", type=int, default=None)

    # Curriculum and GMM Manifold Hyper-Parameters
    p.add_argument("--min_lambda", type=float, default=None)
    p.add_argument("--min_stage_epochs", type=int, default=None)
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--kappa", type=float, default=None)
    p.add_argument("--delta", type=float, default=None)

    args_parsed = p.parse_args()

    # Enforce mandatory parameters manually if not using 'required=True'
    essential_fields = ['backbone', 'loss_type', 'time_enc', 'vocab_size', 'd_model', 'n_layers', 'epochs',
                        'batch_size', 'lr', 'pkt_num']
    for field in essential_fields:
        if getattr(args_parsed, field) is None:
            print(f"CRITICAL ERROR: Argument '--{field}' must be explicitly provided in industrial mode.")
            sys.exit(1)

    SSHCAMTrainer(args_parsed).train()