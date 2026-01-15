import torch
from tqdm import tqdm


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    save_path="model.pt"
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training loop
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1} - Train"
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1} - Val"
            ):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )

    
    # Save trained model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": epochs
        },
        "model.pt"
    )

    return train_losses, val_losses
