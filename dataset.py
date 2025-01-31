import torch
import torchvision
import torchvision.transforms as transforms

# MNIST dataset (images are 28x28)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to mean=0.5, std=0.5
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
val_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)