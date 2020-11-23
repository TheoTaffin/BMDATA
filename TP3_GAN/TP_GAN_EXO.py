# Native libraries
import os

# Data Visualization/Manipulation libraries
import matplotlib.pyplot as plt

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
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


############ Discriminator Network

# The discriminator takes an image as input and ties to classigy it as "real" or "fake". In this
# sense, it's like any other neural network. While we can use a CNN for the discriminator,
# we'll use a simple feedforward network with 3 linear layers to keep things since (it's in the
# jupyter lab, i've no idea what this sentence means). We'll trat each image as a vector of size
# 784 (28*28)

image_size = 28*28
hidden_size = 256

"""
LeakyReLU : different from the regular ReLU function, it allows the pass of a small gradient 
signal for negatives values. As a result, it makes the gradients from the discriminator flow 
stronger into the generator. Instead of passing a gradient (slope) of 0 int the back prop pass, 
it passes a small negative gradient
"""
D = nn.Sequential(
    nn.Linear(in_features=image_size, out_features=hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(in_features=hidden_size, out_features=hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(in_features=hidden_size, out_features=1),
    nn.Sigmoid())
D.to(device)

############ Generator Network
# The input to the generator is typically a vector or a matrix which is used as a seed for
# generating an image. Once again, to keep things simple, we'll use a feedforward neural network
# with 3 layers, and the output will be a vector of size 784, which can be transformed to a
# 28x28 px image.
latent_size = 64  # the number of random values to be generated

"""
The ReLu activation (Nair & Hinton, 2010), is used in the generator with the exception of the 
output layer which uses the Tanh function. We observed that using a bounded activation allowed 
the model to learn more quickly to saturate and cover the color space of the training 
distribution. Within the discriminator we found the leaky rectified activation (Maas et al., 
2013) (Xu et al., 2015) to work well, especially for higher resolution modeling.
"""
G = nn.Sequential(
    nn.Linear(in_features=latent_size, out_features=hidden_size),
    nn.ReLU(),
    nn.Linear(in_features=hidden_size, out_features=hidden_size),
    nn.ReLU(),
    nn.Linear(in_features=hidden_size, out_features=image_size),
    nn.Tanh()
)

# not that since the outputs of the tanH activation lies in the range [-1, 1], we have applied
# the same transformation to the images in the training dataset. let's generate an output vector
# using the generator and view it as an image by transforming and denormalizing the output
y = G(torch.randn(2, latent_size))
gen_imgs = denorm(y.reshape((-1, 28, 28)).detach())
plt.imshow(gen_imgs[0], cmap='gray')
plt.show()
G.to(device)


############ Discriminator training
# Since the discriminator is a binary classification problem, we'll use the binary crossentropy
# loss function to quantify how well it is able to differentiate between real and generated images.

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


def train_discriminator(img_fn):
    # Create the labels which are later used as input for the BCE loss
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # Computes loss for real images
    outputs = D(img_fn)
    d_loss_real = criterion(outputs, target=real_labels)
    real_score = outputs

    # Loss for the fake images
    z = torch.randn((batch_size, latent_size)).to(device)
    fake_images = G(z)
    outputs = D(fake_images)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    # Combine losses
    d_loss = d_loss_fake + d_loss_real
    # Reset gradients
    reset_grad()
    # Computes gradients
    d_loss.backward()
    # Adjust the parameters using backpropagation
    d_optimizer.step()

    return d_loss, real_score, fake_score


############ Discriminator training
# Since the outputs of the generators are images, it's not obvious how we could train it. This
# is where we use the discriminator as a part of the loss function :
# - First, we generate a batch of images using the generator, and pass it to the discriminator
# - Second, we calculate the loss by setting the target labels to 1, i.e real. We do this
# because the generators objective is to "fool" the discriminator
# - Third, we use tghe loss to perform gradient descent, i.e change the weights of the
# generator, so it gets better at generating real-like images.


# note :remember we already implemented that :  "g_optimizer = torch.optim.Adam(G.parameters(),
# lr=0.0002)" it's not great for comprehension but since we're using this as a script..


def train_generator():
    # Generate fake images and calculate loss
    z = torch.randn((batch_size, latent_size)).to(device)
    fake_images = G(z)
    labels = torch.ones(batch_size, 1).to(device)
    g_loss = criterion(D(fake_images), labels)

    # backprop and optimize
    reset_grad()
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images


############ Training the model
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# Saving some real images
for images, _ in data_loader:
    images_conv = images.reshape(images.size(0), 1, 28, 28)
    save_image(denorm(images_conv), os.path.join(sample_dir, 'real_images.png'), nrow=10)
    break


# We'll also define a helper function to save a batch of generated images to disk at the end of
# every epoch. We'll use a fixed set of input vectors to the generator to see how the individual
# generated images evolve over time as we train the model
sample_vectors = torch.randn(batch_size, latent_size).to(device)


def save_fake_images(index):
    fake_images = G(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print(f'Saving: {fake_fname}')
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)


# State of these images before training :
save_fake_images(0)

# We can now train the model. In each epoch, we train the discriminator first and then the
# generator.
num_epochs = 150
total_step = len(data_loader)
d_losses, g_losses, real_scores, fake_scores = [], [], [], []

for epoch in range(num_epochs):
    for count, (images, _) in enumerate(data_loader):
        # Load a batch and transform to vector
        images_conv = images.reshape(-1, 28*28).to(device)

        # Training the discirminator and generator
        d_loss, real_score, fake_score = train_discriminator(images_conv)
        g_loss, fake_images = train_generator()


        # checking on the losses
        if (count+1) % 200 == 0:
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            real_scores.append(real_score.mean().item())
            fake_scores.append(fake_score.mean().item())
            print(
                'Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                .format(epoch, num_epochs, count + 1, total_step, d_loss.item(), g_loss.item(),
                        real_score.mean().item(), fake_score.mean().item()))

    # Sample and save images
    save_fake_images(epoch+1)


# Saving model checkpoint
torch.save(G.state_dict(), "G.ckpt")
torch.save(D.state_dict(), "D.ckpt")


# We can then visualize how the loss changed over time. This is quite useful for debugging the
# training process. For GANs, we expect  the generator's loss to reduce over time without the
# discriminator's loss getting too high
plt.plot(d_losses, '-')
plt.plot(g_losses, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses')

plt.plot(real_scores, '-')
plt.plot(fake_scores, '-')
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend(['Real Score', 'Fake score'])
plt.title('Scores')
