import torch
import matplotlib.pyplot as plt


def evaluate_and_visualize(
    model,
    dataloader,
    dataset,
    device,
    max_images=16
):
    model.eval()

    correct = 0
    total = 0
    failed = []

    class_names = dataset.classes

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i in range(images.size(0)):
                if preds[i] != labels[i]:
                    failed.append({
                        "image": images[i].cpu(),
                        "true": labels[i].item(),
                        "pred": preds[i].item()
                    })

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Failed samples: {len(failed)} / {total}")

    if len(failed) == 0:
        print("No misclassified samples")
        return

    num_show = min(max_images, len(failed))
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_show):
        sample = failed[i]
        img = sample["image"].permute(1, 2, 0)

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(
            f"P: {class_names[sample['pred']]}\n"
            f"T: {class_names[sample['true']]}",
            color="red",
            fontsize=9
        )

    for j in range(num_show, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Misclassified Test Images")
    plt.tight_layout()
    plt.show()