import torch
import torch.nn as nn
import torch.optim as optim

import hydra
from omegaconf import DictConfig
import wandb
from dataset import train_dataset, val_dataset
from model import CustomCNN

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    # initializing wandb
    wandb.init(
        project = cfg.wandb.project,
        entity = cfg.wandb.entity,
        config = cfg.wandb.run_name,
        config = cfg
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )
    model = CustomCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    num_epochs = wandb.config.num_epochs  # or set directly if you don't want to use wandb.config
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log training info every 100 steps
            if (step + 1) % 100 == 0:
                average_loss = running_loss / 100.0
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], Loss: {average_loss:.4f}")

                # Log the current loss, epoch, and step to wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "loss": average_loss
                })

                # Reset running_loss for the next logging interval
                running_loss = 0.0
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # Get the predicted class with the highest score
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")
        # Log metrics to W&B
        wandb.log({"val_accuracy": accuracy})

    # Optionally finish the run
    wandb.finish()

if __name__ == '__main__':
    main()

