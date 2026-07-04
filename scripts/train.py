import os
import sys
import yaml
import argparse
from pathlib import Path
import warnings

# 프로젝트 루트 경로 추가 (scripts/.. 위치)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    parser = argparse.ArgumentParser(description="CXR-CAD Training Script")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default=None, help='Model architecture override')
    parser.add_argument('--fold', type=int, default=1, help='Which fold to train (1-5), default: 1')
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        CFG = yaml.safe_load(f)
        
    NIH_DIR = CFG['data']['data_root']
    NIH_CSV = Path(NIH_DIR) / CFG['data']['metadata_csv'].split('/')[-1]
    IMG_DIR = Path(NIH_DIR) / "images"
    CHECKPOINT_DIR = Path(CFG['train']['checkpoint_dir'])
    
    # 설정 오버라이드
    MODEL_KEY = args.model if args.model else CFG['model']['default']
    IMAGE_SIZE = CFG['data']['image_size']
    BATCH_SIZE = CFG['train']['batch_size']
    EPOCHS = CFG['train']['epochs']
    LR = CFG['train']['lr']
    WD = CFG['train']['weight_decay']
    GAMMA = CFG['train']['focal_gamma']
    N_WORKERS = CFG['train']['num_workers']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 데이터 로드 및 필터링
    df = load_nih_csv(str(Path(NIH_CSV).parent))
    df = df[df['Patient Age'] <= 100].dropna(subset=['Full_Path']).reset_index(drop=True)
    
    train_df, test_df = split_by_patient(
        df, test_ratio=CFG['data']['test_ratio'], random_state=CFG['data']['seed']
    )
    verify_no_leakage(train_df, test_df)
    
    # 2. Fold 분할
    splits = get_group_kfold_splits(train_df, n_splits=CFG['train']['n_splits'])
    train_idx, val_idx = splits[args.fold - 1]
    
    fold_train_df = train_df.iloc[train_idx]
    fold_val_df = train_df.iloc[val_idx]
    pos_weight = compute_pos_weight(fold_train_df)

    # 3. 데이터 로더
    train_loader = create_dataloader(
        fold_train_df, str(IMG_DIR), get_train_transforms(IMAGE_SIZE),
        batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True
    )
    val_loader = create_dataloader(
        fold_val_df, str(IMG_DIR), get_inference_transforms(IMAGE_SIZE),
        batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False
    )

    # 4. 모델 셋업
    model = build_model(MODEL_KEY, pretrained=True).to(device)
    criterion = build_loss(gamma=GAMMA, pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    early_stopping = EarlyStopping(patience=CFG['train']['early_stopping_patience'], mode='max')
    
    ckpt_path = CHECKPOINT_DIR / f"{MODEL_KEY}_fold{args.fold}_best.pth"

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

    # 5. 학습 시작
    print(f"\n[Fold {args.fold}] Start Training {MODEL_KEY}...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        best_auroc, best_auprc, history_df = trainer.fit(
            train_loader, val_loader, EPOCHS, DISEASE_LABELS
        )
    
    print(f"\n[Fold {args.fold} Result] Best AUROC: {best_auroc:.4f}, Best AUPRC: {best_auprc:.4f}")

if __name__ == "__main__":
    main()
