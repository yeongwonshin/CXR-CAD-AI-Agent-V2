# CXR-CAD 팀 협업 가이드 (CONTRIBUTING)

5인 구성(Data, AI, BE, FE)의 원활한 프로젝트 진행을 위한 GitHub 협업 규칙입니다.

> **💡 필독 문서 안내**
> - 본 문서는 GitHub 브랜치 동기화 및 코드 통합에 대한 규칙을 다룹니다.
> - 프로젝트 역할별 실행 로드맵 및 상세한 워크플로우는 [`TEAM_WORKFLOW.md`](TEAM_WORKFLOW.md)를 먼저 확인해 주세요.
> - Kaggle 기반 학습 환경 구축 및 구동 방법은 [`KAGGLE_SETUP.md`](KAGGLE_SETUP.md)를 참고해 주세요.

## 1. 브랜치 전략 (Branching Strategy)

우리는 단순화된 **GitHub Flow** 방식에 파트별 Prefix를 혼합하여 사용합니다.

- `main` 브랜치: 언제나 배포 가능한 프로덕션 코드. (직접 Push 금지 🚫)
- `dev` 브랜치: 기능들이 통합되어 테스트되는 개발 브랜치.
- **기능 브랜치 (Feature Branches)**:
  `{파트}/{기능명}` 형식으로 생성합니다.
  - Data 팀: `data/data-loader`, `data/augmentation`
  - AI 팀: `ai/model-architecture`, `ai/grad-cam`
  - BE 팀: `be/predict-api`, `be/docker-setup`
  - FE 팀: `fe/dashboard-ui`, `fe/api-integration`
  - 픽스(Fix): `fix/{버그명}`

## 2. 머지 규칙 (Merge Rules)

### GitHub Branch Protection Rules 설정 (권장)
GitHub 레포지토리의 `Settings > Branches`에서 `main`과 `dev` 브랜치에 대해 다음 규칙을 켜두는 것을 강력히 권장합니다:
1. **Require pull request reviews before merging**: 최소 **1명 이상의 Approve**가 있어야 Merge 가능하도록 설정. 본인이 작성한 코드는 본인이 Merge할 수 없습니다.
2. **Require status checks to pass before merging**: CI(Test, Lint 등)가 성공해야만 Merge 가능. (추후 GitHub Actions 세팅 시)

### Merge 전략
- **`dev` 브랜치로 기능 브랜치 Merge 할 때**: `Squash and Merge`를 사용합니다. 
  - 이유: 자잘한 커밋 기록(예: "오타 수정", "print문 제거")을 하나로 깔끔하게 압축하여 `dev` 브랜치의 히스토리를 깨끗하게 유지하기 위함입니다.
- **`dev`에서 `main` 브랜치로 배포 준비 시**: 주기적으로 `Create Pull Request`를 띄워 파트 리더급 혹은 팀원 전체 리뷰 후 `Rebase and Merge` 혹은 `Merge Commit`을 생성하여 메인 브랜치로 보냅니다.

## 3. 코드 리뷰 및 협업 포인트

PR을 올릴 때는 미리 만들어둔 **PR 템플릿** 양식에 맞춰 상세히 작성합니다.

- **Data / AI 팀**: 모델의 성능 하락(Degradation)이 없는지, Data Leakage(학습 데이터에 평가 데이터가 섞임)가 발생하지 않는 구조인지 중점 리뷰.
- **BE 팀**: API 응답 지연(Latency)이 심하지 않은지, Pydantic 모델의 에러 핸들링 로직 점검.
- **FE 팀**: UI/UX 깨짐 방지, 각 파트 서버가 죽었을 때 무한 로딩 등에 대한 예외 처리 점검.

### 특별 관리 항목 (중요)

1. **Jupyter Notebook (`.ipynb`) 커밋 규칙**
   - 노트북 파일은 불필요한 로그나 너무 긴 출력 결과물로 인해 PR 리뷰가 어려워지고 Merge Conflict가 자주 발생할 수 있습니다.
   - 따라서 노트북 파일을 커밋할 때는 **불필요한 출력은 제외하고 공유 및 리뷰에 필요한 내용과 결과만 깔끔하게 정리한 후 업로드**해 주세요.
2. **모델 가중치 (`*.pth`) 및 대용량 파일 공유 방식**
   - 용량/보안 문제로 `.gitignore`에 명시되어 `checkpoints/` 하위의 가중치 및 대용량 분석 데이터(`.csv` 등)는 Git에 올라가지 않습니다. (예: `checkpoints/<model>/<model>_best.pth`)
   - 학습이 완료된 모델 가중치나 결과 파일들은 구글 드라이브(Google Drive), 슬랙(Slack) 등 팀 내 **별도의 공유 채널**을 통해 배포해야 합니다.
   - FE/BE 개발자는 전달받은 파일을 로컬 환경의 `checkpoints/<model>/` 서브폴더 내부에 수동으로 배치하여 서버를 구동해 주세요.
3. **단일 진실 공급원 (`config.yaml`) 변경 시 주의**
   - 실험을 통해 얻은 최적의 하이퍼파라미터를 `configs/config.yaml`에 반영할 때는 시스템 전체의 파이프라인에 영향을 미칩니다.
   - 설정 파일의 변경 사항을 올릴 때는 해당 PR에 구체적인 변경 사유(예: "Focal loss 실험을 통한 최적화 결과 gamma=2 적용")를 명시하고 관계된 파트의 확인을 받아주세요.

## 4. 커밋 메시지 컨벤션 (Commit Convention)

커밋 메시지는 작업 내용을 명확히 파악할 수 있도록 [Karma 스타일](https://karma-runner.github.io/6.0/dev/git-commit-msg.html)의 태그를 사용합시다.

- `feat:` 새로운 기능 추가
- `fix:` 버그 수정
- `docs:` 문서 수정 (README, CONTRIBUTING 등)
- `style:` 코드 포맷팅, 세미콜론 누락 등 (코드 로직 변경 없음)
- `refactor:` 코드 리팩토링 (기능 변화 없음)
- `test:` 테스트 코드 작성
- `chore:` 빌드 업무 수정, 패키지 매니저 수정 (.gitignore, requirements.txt 등)

예시: `feat: data_loader.py에 CSV 파서 로직 추가`
