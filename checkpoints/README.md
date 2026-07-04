# checkpoints/

모델별 서브디렉토리에 학습 산출물을 저장합니다.

## 디렉토리 구조

```
checkpoints/
  densenet/
    densenet_best.pth
    test_predictions.csv
    op_analysis.csv
    false_positive.csv   false_negative.csv
    age_subgroup.csv     gender_subgroup.csv   view_subgroup.csv
    shortcut_regions.csv
    domain_shift.csv     domain_shift.png
  efficientnet/
    efficientnet_best.pth
    test_predictions.csv  ...
  vit/
    vit_best.pth
    test_predictions.csv  ...
```

## 파일명 규칙

| 파일 | 설명 |
|------|------|
| `<model>_best.pth` | 최고 val AUROC 체크포인트 |
| `test_predictions.csv` | 테스트셋 예측 확률 |
| `op_analysis.csv` | Operating Point 분석 |
| `domain_shift.csv` | NIH vs CheXpert AUROC 비교 |
| `*_subgroup.csv` | 성별/나이/뷰 Subgroup 분석 |
| `false_*.csv` | FP/FN 오류 사례 |
| `shortcut_regions.csv` | Grad-CAM 활성화 영역 |

## Colab 저장 코드 예시

```python
import pathlib
model_key = "densenet"  # or "efficientnet", "vit"
ckpt_dir = pathlib.Path(f"checkpoints/{model_key}")
ckpt_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    "epoch"              : epoch,
    "model_state_dict"   : model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_auroc"          : best_auroc,
}, ckpt_dir / f"{model_key}_best.pth")
```

**주의:** `.pth` 파일은 `.gitignore`에 의해 Git 저장소에 포함되지 않습니다.
