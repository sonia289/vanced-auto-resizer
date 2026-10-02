# VANCED Ad Auto Resize V1

완성형 광고 MP4의 Burn-in 텍스트와 인물을 보호하며 1:1, 4:5, 16:9로 자동 리사이징하는 내부 MVP입니다.

## 기능
- MP4 단일/Batch 업로드
- PaddleOCR Text Detection 우선 + OpenCV fallback
- YOLO person/object detection 우선 + OpenCV HOG person fallback
- Safe Area / 보호율 / Confidence 계산
- Intelligent Crop / Scale-Reposition / Background Extend 자동 선택
- 움직이는 중요영역 중심점 smoothing
- 1080x1080 / 1080x1350 / 1920x1080 렌더링
- FFmpeg H.264 인코딩 및 원본 오디오 유지
- Preview, 개별 다운로드, Batch ZIP

## macOS 빠른 실행
```bash
brew install ffmpeg python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
같은 사내망에서는 `http://<Mac-mini-IP>:8501` 로 접속합니다.

PaddleOCR/PaddlePaddle 또는 YOLO 가중치 로딩에 실패해도 fallback detector로 앱은 계속 동작합니다. YOLO는 최초 실행 시 가중치 다운로드가 필요할 수 있습니다. 제품 전용 detector는 V1 필수 범위 밖이며 일반 객체를 `other`로 반영합니다.
