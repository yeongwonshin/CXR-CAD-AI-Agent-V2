"""
CXR-CAD 학습 루프 뼈대.

실제 학습은 Google Colab 노트북 (notebooks/04_Training.ipynb)에서 수행합니다.
학습 완료 후 생성된 .pth 파일을 checkpoints/<model_key>/ 디렉토리에 배치하면
API 서버(api/main.py)가 자동으로 로드합니다.

구현 예정 기능:
  - Trainer      : 학습 루프, 검증 루프, 체크포인트 저장
  - EarlyStopping: validation AUROC 기준 조기 종료
"""

from __future__ import annotations

# TODO: Colab 노트북에서 구현 후 이 파일에 이식합니다.
import torch
import torch.nn as nn
import numpy as np
import pathlib
import time
import pandas as pd
from tqdm.auto import tqdm


class EarlyStopping:
    """
    Validation metric 기준 Early Stopping.

    Args:
        patience : 개선 없을 때 허용 에폭 수
        min_delta: 개선으로 인정할 최소 변화량
        mode     : 'max' (AUROC 등) | 'min' (loss 등)
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = None
        self.stop      = False

    def __call__(self, metric: float) -> bool:
        """
        Returns:
            True: 학습 중단, False: 계속
        """
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                return True
        return False


class Trainer:
    """
    CXR-CAD 모델 학습 루프.
    Optuna HPO(Hyperparameter Optimization)를 위해 trial.report 및 Pruning을 지원.

    체크포인트 저장 포맷:
        torch.save({
            "epoch"            : epoch,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auroc"        : best_auroc,
            "val_auprc"        : best_auprc,
        }, "checkpoints/<model_key>/<model_key>_best.pth")
    """

    def __init__(self, model, optimizer, criterion, device, early_stopping=None,
                 scheduler=None, use_amp=True, grad_clip=1.0, 
                 checkpoint_path="checkpoints/densenet/densenet_best.pth"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.checkpoint_path = checkpoint_path
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.best_auroc = 0.0
        self.best_auprc = 0.0
        self.history = []

        import pathlib
        pathlib.Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    def train_one_epoch(self, loader):
        import torch.nn as nn
        from tqdm.auto import tqdm
        self.model.train()
        running_loss = 0.0
        for images, labels in tqdm(loader, desc='  Train', leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
        return running_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader, disease_labels):
        import torch
        import numpy as np
        from tqdm.auto import tqdm
        from src.analysis.evaluation import compute_auroc, compute_auprc

        self.model.eval()
        all_probs, all_targets = [], []
        for images, labels in tqdm(loader, desc='  Eval ', leave=False):
            images = images.to(self.device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())
            
        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        
        auroc_dict = compute_auroc(y_true, y_prob, disease_labels)
        auprc_dict = compute_auprc(y_true, y_prob, disease_labels)
        
        return auroc_dict['macro_avg'], auprc_dict['macro_avg'], y_true, y_prob

    def fit(self, train_loader, val_loader, epochs, disease_labels, optuna_trial=None):
        import time
        import torch

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            loss = self.train_one_epoch(train_loader)
            auroc, auprc, _, _ = self.evaluate(val_loader, disease_labels)
            
            if self.scheduler:
                self.scheduler.step()
                
            elapsed = time.time() - t0
            self.history.append({'epoch': epoch, 'loss': loss, 'auroc': auroc, 'auprc': auprc})
            print(f'  Epoch {epoch:3d}/{epochs} | loss={loss:.4f} | AUROC={auroc:.4f} | AUPRC={auprc:.4f} | {elapsed:.0f}s')

            if auroc > self.best_auroc:
                self.best_auroc = auroc
                self.best_auprc = auprc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auroc': self.best_auroc,
                    'val_auprc': self.best_auprc,
                }, self.checkpoint_path)
                print(f'  ★ Best checkpoint saved → {self.checkpoint_path}')

            if optuna_trial:
                import optuna
                optuna_trial.report(auroc, epoch)
                if optuna_trial.should_prune():
                    raise optuna.TrialPruned()

            if self.early_stopping and self.early_stopping(auroc):
                print(f'  Early stopping at epoch {epoch}')
                break
                
        import pandas as pd
        return self.best_auroc, self.best_auprc, pd.DataFrame(self.history)
