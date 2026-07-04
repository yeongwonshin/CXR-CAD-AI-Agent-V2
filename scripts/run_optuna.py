import os
import sys
import yaml
import argparse
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Optuna require
try:
    import optuna
except ImportError:
    raise ImportError("Optuna is not installed. Please run: pip install optuna")

from src.preprocess.data_loader import (
    load_nih_csv, split_by_patient, verify_no_leakage,
    compute_pos_weight, get_group_kfold_splits, create_dataloader,
)
from src.preprocess.transforms import get_train_transforms, get_inference_transforms
from src.train.models import build_model, DISEASE_LABELS
from src.train.focal_loss import build_loss
from src.train.trainer import EarlyStopping, Trainer

import torch
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser(description="CXR-CAD Hyperparameter Optimization via Optuna")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of optuna trials')
    parser.add_argument('--study_name', type=str, default='cxr_cad_hpo', help='Name of the study')
    return parser.parse_args()

def objective(trial, args, CFG, train_df, IMG_DIR, device):
    """
    Optuna Objective Function
    """
    # 하이퍼파라미터 정의 (Search Space)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    focal_gamma = trial.suggest_float('focal_gamma', 0.0, 3.0, step=0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    IMAGE_SIZE = CFG['data']['image_size']
    EPOCHS = min(CFG['train']['epochs'], 15) # HPO를 위해 epoch 수를 약간 제한할 수 있음
    N_WORKERS = CFG['train']['num_workers']
    MODEL_KEY = CFG['model']['default']

    # Fold1 만 사용해서 빠른 HPO 진행
    splits = get_group_kfold_splits(train_df, n_splits=CFG['train']['n_splits'])
    train_idx, val_idx = splits[0]
    
    fold_train_df = train_df.iloc[train_idx]
    fold_val_df = train_df.iloc[val_idx]
    pos_weight = compute_pos_weight(fold_train_df)

    train_loader = create_dataloader(
        fold_train_df, str(IMG_DIR), get_train_transforms(IMAGE_SIZE),
        batch_size=batch_size, num_workers=N_WORKERS, shuffle=True
    )
    val_loader = create_dataloader(
        fold_val_df, str(IMG_DIR), get_inference_transforms(IMAGE_SIZE),
        batch_size=batch_size, num_workers=N_WORKERS, shuffle=False
    )

    model = build_model(MODEL_KEY, pretrained=True).to(device)
    criterion = build_loss(gamma=focal_gamma, pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 조기 종료 조건 완화 (Optuna Pruner가 주로 관리)
    early_stopping = EarlyStopping(patience=5, mode='max')
    
    ckpt_path = Path("checkpoints") / MODEL_KEY / f"optuna_trial_{trial.number}.pth"

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        early_stopping=early_stopping,
        scheduler=scheduler,
        use_amp=CFG['train'].get('use_amp', True),
        grad_clip=CFG['train'].get('grad_clip', 1.0),
        checkpoint_path=str(ckpt_path)
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        best_auroc, _, _ = trainer.fit(
            train_loader, val_loader, EPOCHS, DISEASE_LABELS, optuna_trial=trial
        )

    # 평가 지표 (최대화 대상)
    return best_auroc

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        CFG = yaml.safe_load(f)
        
    NIH_DIR = CFG['data']['data_root']
    NIH_CSV = Path(NIH_DIR) / "Data_Entry_2017.csv"
    IMG_DIR = Path(NIH_DIR) / "images"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터 로드
    df = load_nih_csv(str(Path(NIH_CSV).parent))
    df = df[df['Patient Age'] <= 100].dropna(subset=['Full_Path']).reset_index(drop=True)
    
    train_df, test_df = split_by_patient(
        df, test_ratio=CFG['data']['test_ratio'], random_state=CFG['data']['seed']
    )

    # Optuna Pruner 설정 (Median Pruner)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        study_name=args.study_name, 
        direction="maximize", 
        pruner=pruner
    )

    print(f"\n[Optuna] Starting optimization with {args.n_trials} trials...")
    study.optimize(lambda trial: objective(trial, args, CFG, train_df, IMG_DIR, device), n_trials=args.n_trials)

    print("\n[Optuna Result] Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(study.get_trials(states=(optuna.trial.TrialState.PRUNED,)))}")
    print(f"  Number of complete trials: {len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))}")

    print("\n[Optuna Result] Best trial:")
    best_trial = study.best_trial
    print(f"  Value (AUROC): {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
