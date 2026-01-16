import torch
import torch.nn as nn

ONNX_OUT = "fruit_model_untrained_dynamicHW.onnx"
NUM_CLASSES = 5
KERNEL_SIZE = 3


def build_model(num_classes=NUM_CLASSES):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=KERNEL_SIZE, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(32, 64, kernel_size=KERNEL_SIZE, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 128, kernel_size=KERNEL_SIZE, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 256, kernel_size=KERNEL_SIZE, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),

        nn.Flatten(),
        nn.Linear(256, num_classes),
    )

def export_untrained_model():
    model = build_model(num_classes=NUM_CLASSES)
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128)

    print(f"Exporting UNTRAINED model to: {ONNX_OUT}")

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUT,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch"},
        },
    )

    print("ONNX saved:", ONNX_OUT)
    print("Supports dynamic image resolution (H x W).")

if __name__ == "__main__":
    export_untrained_model()