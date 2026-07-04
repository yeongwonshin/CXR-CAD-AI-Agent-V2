#!/bin/bash
# Kaggle API를 사용하여 NIH ChestX-ray 데이터셋을 다운로드하고 압축을 해제하는 스크립트

DATA_DIR="data/nih"

# 1. Kaggle API 인증 자격 증명 확인
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Error: ~/.kaggle/kaggle.json 파일이 없습니다."
    echo "진행 방법:"
    echo "1. Kaggle.com -> Account -> 'Create New API Token' 클릭"
    echo "2. 다운로드된 kaggle.json의 내용을 복사"
    echo "3. 서버에서 다음 명령 실행:"
    echo "   mkdir -p ~/.kaggle && nano ~/.kaggle/kaggle.json"
    echo "   (내용 붙여넣기 후 Ctrl+O, Enter, Ctrl+X)"
    echo "   chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "✅ Kaggle API 인증 정보 확인 완료"

# 2. 패키지 설치
echo "📦 Kaggle 라이브러리를 설치합니다..."
pip install -q kaggle

# 3. 디렉토리 준비
mkdir -p "$DATA_DIR/images"
echo "📂 $DATA_DIR 디렉토리 생성 완료"

# 4. 다운로드 및 압축 해제 (NIH ChestX-ray)
echo "⬇️ NIH 데이터셋 다운로드를 시작합니다 (용량이 매우 크니 주의)..."
kaggle datasets download -d nih-chest-xrays/data -p "$DATA_DIR" --unzip
echo "✅ NIH 다운로드 완료 및 images로 정리되었습니다!"

# 5. CheXpert 다운로드 (선택, External Validation 용)
CHEXPERT_DIR="data/chexpert"
mkdir -p "$CHEXPERT_DIR"
echo "⬇️ CheXpert 데이터셋 다운로드를 시작합니다..."
kaggle datasets download -d ashery/chexpert -p "$CHEXPERT_DIR" --unzip
echo "✅ CheXpert 다운로드 완료!"

ls -l data/
