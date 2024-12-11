import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

"""
The torchvision.datasets module contains Dataset objects for many real-world vision data like CIFAR, COCO (full list here). 
In this tutorial, we use the FashionMNIST dataset. Every TorchVision Dataset includes two arguments: 
transform and target_transform to modify the samples and labels respectively.
"""
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

"""
We pass the Dataset as an argument to DataLoader. This wraps an iterable over our dataset, 
and supports automatic batching, sampling, shuffling and multiprocess data loading. 
Here we define a batch size of 64, i.e. each element in the dataloader iterable will return a batch of 64 features and labels.
"""

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Při iteraci přes DataLoader získáme v každé iteraci jednu dávku dat.
# V každé dávce jsou pole X a y, kde X je pole s daty a y je pole s odpovídajícími štítky.



"""
To define a neural network in PyTorch, we create a class that inherits from nn.Module. 
We define the layers of the network in the __init__ function and specify how data will pass 
through the network in the forward function. To accelerate operations in the neural network, we move it to the GPU or MPS if available.
"""

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # This layer flattens the input tensor into a 1D tensor
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),    # linear layer that takes 784 input features and outputs 512 features
            nn.ReLU(),                           # ReLU activation function
            nn.Linear(512, 512),  # linear layer that takes 512 input features and outputs 512 features
            nn.ReLU(),                                # another ReLU activation function
            nn.Linear(512, 10),   # linear layer that takes 512 input features and outputs 10 features
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits  # Logity jsou v podstatě surová, nenormalizovaná skóre vypočítaná poslední vrstvou neuronové sítě.
                       # Představují, jak silně síť "věří" tomu, že daný vstup patří do konkrétní třídy.

model = NeuralNetwork().to(device)
print(model)

"""
To train a model, we need a loss function and an optimizer.
In a single training loop, the model makes predictions on the training dataset (fed to it in batches), 
and backpropagates the prediction error to adjust the model’s parameters.
"""

# Porovnává predikované pravděpodobnosti s pravdivými štítky. Čím větší je rozdíl mezi predikovanými a pravdivými pravděpodobnostmi, tím větší je ztráta.
# Cílem je minimalizovat tuto ztrát
loss_fn = nn.CrossEntropyLoss()

# Stochastic Gradient Descent - Vypočítá gradient ztrátové funkce vzhledem k parametrům modelu.  Aktualizuje parametry
# v opačném směru gradientu, s velikostí kroku danou učící rychlostí (learning rate).
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()  # backwardpass s pocitanim gradientu
        optimizer.step()  # Upraví každý parametr tak, že od něj odečte gradient vynásobený učící rychlostí (learning rate)
        optimizer.zero_grad() # potřeba gradienty vymazat, aby se nekumulovaly s gradienty z následujícího batchu

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
           # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

"""
The training process is conducted over several iterations (epochs). During each epoch, the model learns parameters 
to make better predictions. We print the model’s accuracy and loss at each epoch; we’d like to see the accuracy increase 
and the loss decrease with every epoch.
"""
if __name__ == "__main__":
    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")


    """A common way to save a model is to serialize the internal state dictionary (containing the model parameters)."""
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")









