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
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False
    )
    model = CustomCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    num_epochs = cfg.num_epochs
    # beginning training
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

            # Print training info
            if (step + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
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


    # Finish wandb run
    wandb.finish()

