from ultralytics import YOLO

def main():
    # ==========================================
    # 1. 모델 설정
    # ==========================================
    # yolov8n.pt: 이미 똑똑한 모델(Pre-trained)을 불러옵니다.
    print(" 모델을 불러오는 중입니다...")
    model = YOLO('yolov8n.pt') 

    # ==========================================
    # 2. 학습 시작 (Training)
    # ==========================================
    # data: data.yaml 파일의 위치 (절대 경로를 추천합니다)
    # epochs: 공부 반복 횟수 (50번 정도면 충분)
    # imgsz: 이미지 크기 (640이 국룰)
    # device: 0 (그래픽카드 사용), cpu (느림)
    print("▶ 학습을 시작합니다! (이 과정은 시간이 걸립니다)")
    results = model.train(
        data='../../dataset/data.yaml',  # 상대 경로 (상황에 맞게 수정 필요)
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,      # 성능이 안 오르면 10번 기다리다 조기 종료
        device=0,         # GPU 사용 (RTX 5060)
        project='A-PAS_Project',
        name='crosswalk_model',
        exist_ok=True
    )
    print("✅ 학습 완료!")

    # ==========================================
    # 3. 라즈베리 파이용 변환 (Export)
    # ==========================================
    # 학습된 최고 성능 모델(best.pt)을 Coral TPU용(tflite)으로 변환합니다.
    print("▶ Coral TPU용으로 변환을 시작합니다...")
    
    # 학습된 모델 다시 불러오기
    best_model = YOLO('A-PAS_Project/crosswalk_model/weights/best.pt')
    
    # Edge TPU 포맷으로 내보내기 (int8 양자화 포함)
    best_model.export(format='edgetpu')
    
    print("🎉 모든 작업이 끝났습니다!")
    print("생성된 'best_full_integer_quant_edgetpu.tflite' 파일을 라즈베리 파이로 옮기세요.")

if __name__ == '__main__':
    main()