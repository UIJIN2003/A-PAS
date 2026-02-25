import torch
import torch.nn as nn

# 1. 모델 클래스 정의 (현재까지 쓰던 것과 동일)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 20) # PRED_LENGTH=10 * 2

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 2. 양자화 및 로드
model = LSTMModel(2, 128, 2)
model.load_state_dict(torch.load("models/best_trajectory_model.pth", map_location='cpu'))
model.eval()

# 동적 양자화 적용
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)

# 3. TorchScript 변환 (Tracing)
example_input = torch.rand(1, 10, 2) # (Batch, Seq, Feature)
traced_model = torch.jit.trace(quantized_model, example_input)

# 4. 최종 라파 전용 파일 저장
traced_model.save("models/best_model_pi.pt")
print("✅ 최적화 완료: models/best_model_pi.pt 파일이 생성되었습니다!")