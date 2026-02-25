import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque
from gpiozero import LED  # ✅ LED 제어 라이브러리 추가

# ==========================================
# ⚙️ [설정] 하드웨어 및 모델 설정
# ==========================================
IMG_W, IMG_H = 1920, 1080 
SEQ_LENGTH = 10 
PRED_LENGTH = 10 
HIDDEN_SIZE = 128
NUM_LAYERS = 2

YOLO_MODEL_PATH = "yolov8s.pt"
LSTM_MODEL_PATH = "models/best_model_pi.pt"
VIDEO_PATH = "my_video.mp4" 

# 하드웨어 설정
alert_led = LED(18)  # ✅ GPIO 18번 핀에 LED 연결
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 🧠 [모델 정의]
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, PRED_LENGTH * 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(-1, PRED_LENGTH, 2)

# 모델 로드
print("📦 모델 로드 중...")
yolo_model = YOLO(YOLO_MODEL_PATH)
lstm_model = LSTMModel(2, HIDDEN_SIZE, NUM_LAYERS).to(device)
lstm_model = torch.jit.load(LSTM_MODEL_PATH, map_location=device)
lstm_model.eval()

track_history = {}
last_predictions = {}
cap = cv2.VideoCapture(VIDEO_PATH)

# ==========================================
# 🎬 [실행] 메인 루프
# ==========================================
print("🚀 A-PAS 모니터링 시작!")

try: # ✅ 예외 처리를 통해 안전한 종료 보장
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.resize(frame, (IMG_W, IMG_H))
        
        # imgsz=320으로 설정하여 라즈베리 파이 부하 감소
        results = yolo_model.track(frame, persist=True, imgsz=320, conf=0.2, verbose=False)
        collision_risk = False
        all_current_preds = {}

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                curr_pos = np.array([float(x), float(y)])
                
                # 1. 이상 행동(Anomaly) 시각화
                if track_id in last_predictions:
                    pred_pos = last_predictions[track_id]
                    anomaly_loss = np.linalg.norm(curr_pos - pred_pos)
                    if anomaly_loss > 50:
                        cv2.putText(frame, f"UNSTABLE: {int(anomaly_loss)}", (int(x), int(y)-40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                if track_id not in track_history:
                    track_history[track_id] = deque(maxlen=SEQ_LENGTH)
                track_history[track_id].append(curr_pos)

                # 2. 미래 경로 예측
                if len(track_history[track_id]) == SEQ_LENGTH:
                    input_seq = np.array(list(track_history[track_id]))
                    input_seq[:, 0] /= IMG_W
                    input_seq[:, 1] /= IMG_H
                    input_seq_torch = torch.FloatTensor([input_seq]).to(device)
                    
                    with torch.no_grad():
                        pred = lstm_model(input_seq_torch).cpu().numpy()[0]
                        pred[:, 0] *= IMG_W
                        pred[:, 1] *= IMG_H
                        all_current_preds[track_id] = pred
                        last_predictions[track_id] = pred[0] 

                        # 예측 경로 시각화
                        cv2.polylines(frame, [pred.astype(np.int32)], False, (0, 0, 255), 2)

            # 3. 충돌 감지 로직
            for id1, pred1 in all_current_preds.items():
                for id2, pred2 in all_current_preds.items():
                    if id1 >= id2: continue 
                    dist = np.linalg.norm(pred1[-1] - pred2[-1])
                    if dist < 100: 
                        collision_risk = True
                        cv2.line(frame, tuple(pred1[-1].astype(int)), tuple(pred2[-1].astype(int)), (0, 255, 255), 3)

        # ==========================================
        # 💡 [핵심] 하드웨어 피드백 (LED 제어)
        # ==========================================
        if collision_risk:
            alert_led.on()  # 🔴 위험 시 LED 점등
            cv2.rectangle(frame, (0, 0), (IMG_W, IMG_H), (0, 0, 255), 20)
            cv2.putText(frame, "!!! COLLISION WARNING !!!", (IMG_W//2-350, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)
        else:
            alert_led.off() # 🟢 안전 시 LED 소등

        # 결과 창 출력 (SSH 환경에서는 주석 처리 필요할 수 있음)
        cv2.imshow("A-PAS Final Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

finally: # ✅ 프로그램 종료 시 반드시 실행
    print("\nCleaning up...")
    alert_led.off() # LED 끄기
    cap.release()
    cv2.destroyAllWindows()