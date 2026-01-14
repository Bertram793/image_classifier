import torch
import torch.nn as nn

# ---- Your model architecture ----
kernel_size = 3
NUM_CLASSES = 5
x_size = 128
y_size = 128

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=kernel_size, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(32, 64, kernel_size=kernel_size, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(64, 128, kernel_size=kernel_size, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(128, NUM_CLASSES),
)

model.eval()

# ---- Dummy input ----
dummy = torch.randn(1, 3, x_size, y_size)

# ---- Export ----
onnx_path = "fruit_model.onnx"
torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=12
)

print(f"✅ Exported ONNX file: {onnx_path}")