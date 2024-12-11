
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from app import NeuralNetwork

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

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

"""
The process for loading a model includes re-creating the model structure and loading the state dictionary into it.
"""
modelTrained = NeuralNetwork().to(device)
modelTrained.load_state_dict(torch.load("model.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

F, T = 0,0
modelTrained.eval()
for i in range(100):
    x, y = test_data[i][0], test_data[i][1]
    with torch.no_grad():
        x = x.to(device)
        pred = modelTrained(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        if predicted != actual:
            F += 1
        else :
            T += 1

print(f"Wrong predictions: {F}, True predictions: {T}")