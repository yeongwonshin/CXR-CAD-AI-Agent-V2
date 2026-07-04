# Technical Guide

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Run the API server

```bash
uvicorn api.main:app --reload --port 8000
```

Open Swagger UI:

```text
http://localhost:8000/docs
```

## 3. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Open the dashboard:

```text
http://localhost:8501
```

## 4. Docker run

GPU access is recommended for real model inference.

```bash
docker compose up --build
```

Detached mode:

```bash
docker compose up -d --build
```

## 5. API endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Service health, API version, device, loaded models, and placeholder state |
| `/models` | GET | Supported inference modes and model metadata |
| `/predict` | POST | Single-image or DICOM prediction with report and agent metadata |
| `/agent/status` | GET | Agent runtime and LLM configuration status |
| `/agent/analyze` | POST | Multi-image agent workflow |
| `/agent/chat` | POST | Follow-up Q&A over an existing agent result |
| `/feedback` | POST | Save clinician feedback into a JSONL review queue |
| `/feedback/queue` | GET | Read recent feedback queue items |

## 6. Example API calls

Health check:

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

List models:

```bash
curl -s http://localhost:8000/models | python -m json.tool
```

Single prediction:

```bash
curl -X POST "http://localhost:8000/predict?model=ensemble&threshold=0.3" \
  -F "file=@sample_cxr.png" | python -m json.tool
```

Agent batch analysis:

```bash
curl -X POST "http://localhost:8000/agent/analyze?model=ensemble&threshold=0.3&question=Which%20case%20should%20be%20reviewed%20first%3F" \
  -F "files=@case_1.png" \
  -F "files=@case_2.dcm" | python -m json.tool
```

## 7. Checkpoint layout

Model artifacts are stored by model key.

```text
checkpoints/
  densenet/
    densenet_best.pth
    test_predictions.csv
    op_analysis.csv
    false_positive.csv
    false_negative.csv
    age_subgroup.csv
    gender_subgroup.csv
    view_subgroup.csv
    shortcut_regions.csv
    domain_shift.csv
    domain_shift.png
  efficientnet/
    efficientnet_best.pth
    ...
  vit/
    vit_best.pth
    ...
```

The API searches for checkpoints in this order:

```text
checkpoints/<model_key>/<model_key>_best.pth
checkpoints/<model_key>/*.pth
checkpoints/<model_key>_best.pth
```

## 8. Checkpoint save format

Recommended format:

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_auroc": best_auroc,
}, "checkpoints/<model_key>/<model_key>_best.pth")
```

The loader supports `model_state_dict`, `state_dict`, and direct state-dict checkpoint formats.

## 9. Additional model weights

Large `.pth` checkpoint files are not committed to Git. The current project README pointed to the following external folder for additional trained model weight files:

```text
https://drive.google.com/drive/folders/1plw8yetMsk07lFYJAjxJjMwjel4-d-IQ?usp=drive_link
```

## 10. Training and validation workflow

Recommended hybrid workflow:

1. Use Vast.ai for heavy training and Optuna hyperparameter search.
2. Use Kaggle for lightweight validation, visualization, and notebook-based analysis.
3. Save best checkpoint files into `checkpoints/<model_key>/`.
4. Upload or mount result CSV files for dashboard analysis.
5. Restart the API so checkpoint detection runs during startup.

Common commands:

```bash
bash scripts/download_data.sh
python scripts/run_optuna.py --n_trials 50
python scripts/train.py --fold 1
pytest tests/ -v
```

For environment-specific details, see the existing `VASTAI_SETUP.md`, `KAGGLE_SETUP.md`, and `TEAM_WORKFLOW.md` files in the project root.

## 11. Repository structure

```text
api/
  main.py                 # FastAPI app, model registry, prediction, agent, feedback endpoints
  schemas.py              # Pydantic request and response models

dashboard/
  app.py                  # Main Streamlit prediction dashboard
  pages/
    agent_workbench.py    # Multi-image agent review page
    analysis_results.py   # Evaluation artifact dashboard
    reliability_readiness.py # Readiness status dashboard
  services/
    llm_analysis.py       # Optional LLM summaries for metrics

src/
  preprocess/             # image, DICOM, split, transform, and data loading utilities
  train/                  # model definitions, focal loss, ensemble, trainer
  analysis/               # evaluation, calibration, Grad-CAM, subgroup, external validation, error analysis
  agentic/                # case tools, dynamic agent, LLM-backed follow-up chat
  reliability/            # readiness report, ROI consistency, hidden stratification

notebooks/                # EDA, preprocessing, training, threshold, calibration, subgroup, external validation, error notebooks
scripts/                  # data download, Optuna, and training entry points
checkpoints/              # model weights and result artifacts by model key
tests/                    # API, encoding, transform, and reliability tests
```

## 12. Tech stack

| Category | Technology |
| --- | --- |
| ML framework | PyTorch, torchvision, timm |
| Model families | DenseNet-121, EfficientNet-B4, ViT-B/16, soft-voting ensemble |
| Optimization | Optuna |
| Preprocessing | Pillow, OpenCV CLAHE, pydicom, albumentations |
| Evaluation | scikit-learn, scipy |
| Explainability | Grad-CAM-style heatmap workflow |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit, Plotly |
| LLM integration | OpenAI-compatible API, LangChain/OpenAI client support |
| Infrastructure | Docker, CUDA runtime, Vast.ai, Kaggle API |
| Testing | pytest, pytest-asyncio, httpx |
| Configuration | YAML and `.env` environment variables |

## 13. Environment variables

Important variables include:

```text
CHECKPOINT_DIR
FEEDBACK_QUEUE_PATH
API_URL
OPENAI_API_KEY
OPENAI_BASE_URL
CXR_AGENT_LLM_MODEL
CXR_AGENT_LLM_ENABLED
CXR_AGENT_LLM_FIRST
CXR_AGENT_TOOL_FIRST_FASTPATH
CXR_AGENT_DEFAULT_FULL_WORKUP
CXR_AGENT_LLM_MAX_TOKENS
CXR_AGENT_LLM_TIMEOUT
CXR_AGENT_LLM_TEMPERATURE
CXR_ANALYSIS_LLM_MAX_TOKENS
CXR_ANALYSIS_LLM_TIMEOUT
CXR_ANALYSIS_LLM_TEMPERATURE
```

Use `.env.example` as the template and keep real keys out of Git.

## 14. Validation checklist

Before a demo or submission, verify:

- API starts and `/health` returns successfully.
- Dashboard can connect to the API.
- The selected inference mode is visible.
- Placeholder mode is clearly shown when checkpoints are absent.
- Real `.pth` files load when placed under `checkpoints/<model_key>/`.
- PNG/JPEG and DICOM inputs can be parsed.
- `/predict` returns probabilities, detected labels, report draft, and metadata.
- Agent Workbench can analyze multiple files.
- Follow-up chat works with either LLM or deterministic fallback.
- Feedback submission writes to the JSONL queue.
- Analysis Results can read checkpoint CSV artifacts.
- Reliability Readiness status responds to threshold changes.
