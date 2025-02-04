import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Batteries-Included PyTorch Walkthrough (MNIST)""")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## First let's install our packages.
        You may have to run the next cell twice. This is an unresolved issue.
        """
    )
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms

    import matplotlib.pyplot as plt
    import marimo as mo
    import wandb
    return F, mo, nn, optim, plt, torch, torchvision, transforms, wandb


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Next, let's set up our PyTorch datasets.

        We'll be using data from the tried and true MNIST dataset, a dataset for testing the ability to read handwritten digits.
        """
    )
    return


@app.cell
def _(torchvision, transforms):
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
    return train_dataset, transform, val_dataset


@app.cell
def _(mo):
    mo.md(r"""## What do samples from our training data look like?""")
    return


@app.cell
def _(plt, train_dataset):
    # Get some samples
    fig, axes = plt.subplots(1, 8, figsize=(10, 5))  # Display 5 images

    for i in range(8):
        image, label = train_dataset[i]  # Get image and label
        axes[i].imshow(image.squeeze(), cmap='gray')  # Remove extra dimension and display in grayscale
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')

    plt.show()
    return axes, fig, i, image, label


@app.cell
def _(mo):
    mo.md(r"""## Sweet, let's set up our dataloaders!""")
    return


@app.cell
def _(torch, train_dataset, val_dataset):
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=64, shuffle=False
    )
    return train_loader, val_loader


@app.cell
def _(mo):
    mo.md(
        """
        ## Next, let's create our model.
        ### Let's try using a custom convolutional neural network 😎
        """
    )
    return


@app.cell
def _(F, nn):
    class CustomCNN(nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
            self.fc2 = nn.Linear(in_features=128, out_features=10)

        def forward(self, x):
            # Convolution + ReLU + MaxPool (1st layer)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            # Convolution + ReLU + MaxPool (2nd layer)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            # Flatten the output for fully connected layers
            x = x.view(-1, 64 * 7 * 7)
            # Fully connected layers
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    return (CustomCNN,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Here's what the diagram looks like, drawn using ASCII characters 😎

        Input
           (1 x 28 x 28)
                 │
                 ▼
        ┌─────────────────┐
        │   Conv2d        │      new dimensions = (input_dim + 2*padding - kernel_size)/stride + 1
        │ (in: 1, out:32) │      new_dimensions = (28 + 2*1 - 3)/1 + 1 = 28    
        │ kernel:3, pad:1 │      new_dimensions = 32 x 28 x 28
        └─────────────────┘
                 │
                 ▼
              ReLU               new_dimensions = 32 x 28 x 28
                 │
                 ▼
        ┌─────────────────┐
        │  MaxPool2d      │      new_dimensions = (input_dim - kernel_size)/stride + 1
        │ (kernel:2, stride:2)   new_dimensions = (28 - 2)/2 + 1 = 14
        └─────────────────┘      new_dimensions = 32 x 14 x 14
                 │
                 ▼
           (32 x 14 x 14)
                 │
                 ▼
        ┌─────────────────┐      new_dimensions = (input_dim +2*padding - kernel_size)/stride + 1
        │   Conv2d        │      new_dimensions = (14 + 2*1 - 3)/1 + 1
        │ (in:32, out:64) │      new_dimensions = 64 x 14 x 14
        │ kernel:3, pad:1 │
        └─────────────────┘
                 │
                 ▼
              ReLU               new_dimensions = 64 x 14 x 14
                 │
                 ▼
        ┌─────────────────┐      new_dimensions = (input_dim - kernel_size)/stride + 1
        │  MaxPool2d      │      new_dimensions = (14 - 2)/2 + 1
        │ (kernel:2, stride:2)   new_dimensions = 64 x 7 x 7
        └─────────────────┘
                 │
                 ▼
           (64 x 7 x 7)
                 │
                 ▼
             Flatten
        (64 * 7 * 7 = 3136)      new_dimensions = 64 * 7 * 7 = 3136
                 │
                 ▼
        ┌─────────────────┐
        │   Linear        │
        │ (3136 -> 128)   │      new_dimensions = 128
        └─────────────────┘
                 │
                 ▼
              ReLU               new_dimensions = 128
                 │
                 ▼
        ┌─────────────────┐
        │   Linear        │      new_dimensions = 10
        │  (128 -> 10)    │
        └─────────────────┘
                 │
                 ▼
               Output
                 (10)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Next, we need to use an appropriate loss function.

        In classification problems, the cross entropy loss is a common choice for measuring the difference between the predicted probability distribution and the true distribution.

        Assume we have:

        - \( C \) classes.
        - A true label represented as a one-hot vector
          $\mathbf{y} = [y_1, y_2, \dots, y_C]$ where $y_i = 1$ if the true class is $i$ and 0 elsewhere.

        - In the example of classifying digits, the number 4 could be represented by [0,0,0,0,1,0,0,0,0,0,0].

        ## Definition of cross-entropy loss

        The cross entropy loss $L$ is defined as:

        $L = - \sum_{i=1}^{C} y_i \log (\hat{y}_i)$

        Because the true label is a one-hot vector, only the log probability of the correct class contributes to the loss. Let $t$ be the index of the true class, then:

        $L = - \log (\hat{y}_t)$


        ## Extension to a batch of examples


        For a batch of $N$ examples, with the $n$-th example having true labels $\mathbf{y}^{(n)}$ and predicted probabilities $\hat{\mathbf{y}}^{(n)}$, the average cross entropy loss is given by:

        $\bar{L} = - \frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^{C} y_i^{(n)} \log \left(\hat{y}_i^{(n)}\right)$

        ## Interpretation

        Minimizing the cross entropy loss encourages the predicted probability for the correct class to be as high as possible, thereby aligning the model's predictions with the true labels.
        """
    )
    return


@app.cell
def _(nn):
    criterion = nn.CrossEntropyLoss()
    return (criterion,)


@app.cell
def _(mo):
    mo.md("""## Let's check out what kind of hardware we're working with 😎""")
    return


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    return (device,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Oh no we're on a CPU 🥴
        No money for local GPU 🥲
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## It's okay let's put our model on a CPU anyway 🥲""")
    return


@app.cell
def _(CustomCNN, device):
    model = CustomCNN().to(device)
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""## Next, let's choose an optimizer. A common starting point is Adam.""")
    return


@app.cell
def _(model, optim):
    optimizer = optim.Adam(model.parameters(), lr=.0001)
    return (optimizer,)


@app.cell
def _(mo):
    mo.md(r"""## Now, we should set up our wandb so we can compare this run against future runs. 😎""")
    return


@app.cell
def _(wandb):
    wandb.init(project='pytorch_tutorial', \
               entity='cbrain', \
               name = 'demo_run_01', \
               config={
                       'num_epochs': 3, \
                       'learning_rate': 0.001,
                        # Add more hyperparameters as needed
                       })
    return


@app.cell
def _(mo):
    mo.md("""## Time to train our model!""")
    return


@app.cell
def _(
    criterion,
    device,
    model,
    optimizer,
    torch,
    train_loader,
    val_loader,
    wandb,
):
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
    return (
        accuracy,
        average_loss,
        correct,
        epoch,
        images,
        labels,
        loss,
        num_epochs,
        outputs,
        predicted,
        running_loss,
        step,
        total,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
