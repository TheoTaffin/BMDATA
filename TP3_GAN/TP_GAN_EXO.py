
# Data Visualization libraries
import matplotlib.pyplot as plt


# PyTorch libraries
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST


# Extraction of the mnist image (0-9 handwritten digits). We normalize the pixle form (0,
# 1) to (-1, 1). reason unclear atm, will be obvious later
mnist = MNIST(root='data', train=True, download=True,
              transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))

# Looking up at a sample tensor from the data. The image and their corresponding label are
# available from the mnist database
img, label = mnist[0]
print(f"label: {label}")
print(img[:, 10:15, 10:15])
print(torch.min(img), torch.max(img))


# since the pixel values are between -1 and 1, we create a denormalization function to be able
# to visualize the data properly (or at all fort that matter, doubt any library can output that)
def denorm(x):
    # x -> [-1; 1] -> [0; 2] -> [0: 1]
    out = (x + 1) / 2
    # in case value are close to 0 or 1 I guess
    return out.clamp(0, 1)


img_norm = denorm(img)
# img_norm[0] bcs the image dim is (1, 28, 28)
plt.imshow(img_norm[0], cmap='gray')
print(f"label: {label}")
plt.show()


# Create a data loader to load the images in batches
batch_size = 100
data_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

for img_batch, label_batch in data_loader:
    print(f"first batch: {img_batch.shape}")
    plt.imshow(img_batch[0][0], cmap='gray')
    print(f"label: {label_batch[0].item()}")
    # if You remove the .item() function you will extract a 1x1 Tensor containing the label
    # value -> No, that's said in the jupyter notebook, no idea why
    break  # this stops for loop


# We are also creating a device which can be used to move the data and models to a GPU,
# if one is available :
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
