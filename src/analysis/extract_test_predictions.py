import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import src.train.models as model_defs
from src.train.models import build_model, DISEASE_LABELS
from src.preprocess.data_loader import build_dataloaders, load_nih_csv, create_dataloader
from src.preprocess.transforms import get_inference_transforms

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 모델 로드
    model_key = args.model
    # Current ViT checkpoints were trained with the torchvision ViT wrapper
    # (state_dict keys such as backbone.class_token). Keep extraction aligned
    # even when timm is installed in the local environment.
    if model_key == "vit":
        model_defs._TIMM_AVAILABLE = False
    model = build_model(model_key)
    ckpt_path = os.path.join(args.checkpoint_dir, model_key, f"{model_key}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)
    model.eval()

    # 2. 데이터셋 및 DataLoader 준비
    # data_loader.py의 build_dataloaders를 사용하되 메타데이터가 필요하므로 직접 구성
    print("Loading NIH metadata...")
    df = load_nih_csv(args.data_dir)
    # 기존 split 로직(Patient ID 기준)을 그대로 따름
    from src.preprocess.data_loader import split_by_patient
    _, test_df = split_by_patient(df, test_ratio=0.15)
    if args.limit:
        test_df = test_df.head(args.limit).reset_index(drop=True)
    
    test_loader = create_dataloader(
        df=test_df,
        images_dir=os.path.join(args.data_dir, "images"),
        transform=get_inference_transforms(224), # 기본 사이즈
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        return_meta=True  # 이미지 이름(Index)을 가져오기 위해 필수
    )
    
    # 3. 메타데이터 매핑용 딕셔너리 (Patient Info)
    meta_dict = test_df.set_index("Image Index").to_dict("index")

    results = []

    # 4. 추론 루프
    print(f"Evaluating {model_key} on the test set ({len(test_df)} images)...")
    with torch.no_grad():
        for images, labels, meta in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            # 예측 로직 (Logits -> Sigmoid)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = labels.cpu().numpy()
            
            image_names = meta["image_index"]
            for i in range(len(image_names)):
                img_name = image_names[i]
                prob = probs[i]
                true_label = labels[i]
                
                info = meta_dict.get(img_name, {})
                row = {
                    "Image Index": img_name,
                    "Patient Age": info.get("Patient Age", np.nan),
                    "Patient Gender": info.get("Patient Gender", "Unknown"),
                    "View Position": info.get("View Position", "Unknown")
                }
                
                # 정답(GT) 저장
                for j, cls in enumerate(DISEASE_LABELS):
                    row[f"{cls}_true"] = float(true_label[j])
                    
                # 예측 확률(Prob) 저장
                for j, cls in enumerate(DISEASE_LABELS):
                    row[f"{cls}_prob"] = float(prob[j])
                    
                results.append(row)

    # 5. CSV 저장
    out_df = pd.DataFrame(results)
    out_path = os.path.join(args.checkpoint_dir, model_key, args.output_name)
    out_df.to_csv(out_path, index=False)
    print(f"\n✅ Successfully saved predictions to {out_path}")
    print(f"이 파일을 로컬 컴퓨터의 checkpoints/{model_key}/ 폴더로 복사한 뒤, 로컬에서 Jupyter Notebook 들을 실행하세요!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="densenet", choices=["densenet", "efficientnet", "vit"])
    parser.add_argument("--data_dir", type=str, default="../data/nih", help="Path to Data_Entry_2017.csv and images")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="Path to best.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test row limit")
    parser.add_argument("--output_name", type=str, default="test_predictions.csv")
    args = parser.parse_args()
    
    main(args)
