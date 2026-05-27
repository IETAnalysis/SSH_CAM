import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import numpy as np

from model import SSHCAMTransformer
from dataset import TrainDataset, EvaluationDataset
from utils import AdaptiveCurriculumScheduler, setup_logger


class SSHCAMTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger(args.save_path)
        self.writer = SummaryWriter(log_dir=args.save_path)

        # 核心：加载并预处理 raw_data
        self.train_data, self.val_data = self._prepare_data(args.train_file, args.eval_file)

        self.model = SSHCAMTransformer(
            len_vocab=args.len_vocab, d_model=args.d_model, nhead=args.nhead,
            n_layers=args.n_layers, num_classes=args.num_classes,
            dropout=args.dropout, kappa=args.kappa, delta=args.delta
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        lams = [round(x, 2) for x in torch.arange(1.0, args.min_lambda - 0.01, -0.05).tolist()]
        self.scheduler = AdaptiveCurriculumScheduler(lams, args.min_stage_epochs, args.epsilon)

    def _prepare_data(self, train_path, eval_path):
        self.logger.info(f"Loading data from {train_path} and {eval_path}...")

        with open(train_path, 'r') as f:
            raw_train = json.load(f)

        train_set = {
            'length': np.array([item['length'] for item in raw_train]),
            'time': np.array([item['time'] for item in raw_train]),
            'dir': np.array([item['dir'] for item in raw_train]),
            'label': np.array([item['label'] for item in raw_train])
        }

        with open(eval_path, 'r') as f:
            raw_eval = json.load(f)

        val_set = {
            'length': np.array([item['length'] for item in raw_eval]),
            'time': np.array([item['time'] for item in raw_eval]),
            'dir': np.array([item['dir'] for item in raw_eval]),
            'label_a': np.array([item['label_a'] for item in raw_eval]),
            'label_b': np.array([item['label_b'] for item in raw_eval]),
            'lam': np.array([item['lam'] for item in raw_eval])
        }

        return train_set, val_set

    def compute_loss(self, z, ya, yb, lam, beta):
        dist = torch.sum(z ** 2, dim=1, keepdim=True) - 2 * torch.matmul(z, self.model.means.T) + \
               torch.sum(self.model.means ** 2, dim=1, keepdim=True).T

        def masked_ce(labels):
            logits = -0.5 * dist
            mask = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), self.args.alpha)
            return F.cross_entropy(logits * (1.0 + mask), labels, reduction='none')

        l_dis = torch.mean(lam * masked_ce(ya) + (1 - lam) * masked_ce(yb))

        da = torch.gather(dist, 1, ya.unsqueeze(1)).squeeze()
        db = torch.gather(dist, 1, yb.unsqueeze(1)).squeeze()
        l_reg = 0.5 * torch.mean(lam * da + (1 - lam) * db)

        return l_dis + beta * l_reg, l_dis, l_reg

    def train(self):
        step = 0

        eval_loader = DataLoader(
            EvaluationDataset(self.val_data),
            batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.workers
        )

        for epoch in range(self.args.epochs):
            curr_lam = self.scheduler.current_lambda()
            train_loader = DataLoader(
                TrainDataset(self.train_data, self.args.pkt_num, curr_lam),
                batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers
            )

            self.model.train()
            beta = self.args.eta * self.scheduler.epoch_cnt

            pbar = tqdm(train_loader, desc=f"Epoch {epoch} [λ={curr_lam}]")
            epoch_loss = 0
            for l, t, d, ya, yb, lv in pbar:
                l, t, d, ya, yb, lv = [x.to(self.device) for x in [l, t, d, ya, yb, lv]]

                z = self.model(l, t, d, (l == 0))
                loss, ld, lr = self.compute_loss(z, ya, yb, lv, beta)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                if step % 50 == 0:
                    self.writer.add_scalar("Loss/Total", loss.item(), step)
                    self.writer.add_scalar("Loss/Dis", ld.item(), step)
                    self.writer.add_scalar("Loss/Reg", lr.item(), step)
                step += 1

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for l, t, d, ya, yb, lv in eval_loader:
                    l, t, d, ya, yb, lv = [x.to(self.device) for x in [l, t, d, ya, yb, lv]]
                    z = self.model(l, t, d, (l == 0))
                    loss, _, _ = self.compute_loss(z, ya, yb, lv, beta)
                    val_loss += loss.item()

            if len(eval_loader) > 0:
                self.writer.add_scalar("Loss/Val", val_loss / len(eval_loader), epoch)

            avg_loss = epoch_loss / len(train_loader)
            if self.scheduler.update(avg_loss):
                self.logger.info(f"Course Advancement -> New λ: {self.scheduler.current_lambda()}")
                torch.save(self.model.state_dict(), os.path.join(self.args.save_path, f"model_lam_{curr_lam}.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, required=True)
    p.add_argument("--save_path", type=str, default="./outputs")
    p.add_argument("--len_vocab", type=int, default=1515)
    p.add_argument("--pkt_num", type=int, default=64)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)

    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)

    p.add_argument("--min_lambda", type=float, default=0.25)
    p.add_argument("--min_stage_epochs", type=int, default=20)
    p.add_argument("--epsilon", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--eta", type=float, default=0.01)
    p.add_argument("--kappa", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=1e-6)

    trainer = SSHCAMTrainer(p.parse_args())
    trainer.train()
