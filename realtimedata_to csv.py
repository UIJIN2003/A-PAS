import cv2
import csv
import os
import time
from ultralytics import YOLO

# ==========================================
# ⚙️ [설정] 실시간 스트림 및 저장 경로
# ==========================================
# 실시간 CCTV 주소 (RTSP 등) 또는 0 (웹캠)
CCTV_SOURCE = "rtsp://username:password@ip_address:port/stream" 
OUTPUT_DIR = "data/RealTime_Collection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "yolov8n.pt"
TARGET_CLASSES = [0, 1, 2, 3, 5, 7] # 보행자, 자전거, 차량 등

def collect_realtime_data():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CCTV_SOURCE)
    
    # 저장할 파일명 (현재 시간 기준)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(OUTPUT_DIR, f"rt_collection_{timestamp}.csv")
    
    with open(csv_file, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['frame', 'track_id', 'class_id', 'x_center', 'y_center', 'width', 'height'])
        
        frame_idx = 0
        print(f"📡 실시간 데이터 수집 시작: {csv_file}")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            frame_idx += 1
            # 실시간성을 위해 persist=True 사용
            results = model.track(frame, persist=True, verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                
                for box, tid, cid in zip(boxes, track_ids, class_ids):
                    if cid in TARGET_CLASSES:
                        x, y, w, h = box
                        wr.writerow([frame_idx, tid, cid, round(x, 2), round(y, 2), round(w, 2), round(h, 2)])
            
            # 화면 표시 (선택 사항)
            annotated_frame = results[0].plot()
            cv2.imshow("CCTV Data Collection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 데이터 수집이 종료되었습니다.")

if __name__ == "__main__":
    collect_realtime_data()