import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def denorm (x):
    out = x * 0.5 + 0.5
    return out.clamp (0, 1)

def imshow(img, title=None):
    """ a function to show tensor images """
    img = img.numpy().transpose(1, 2, 0)
    mean = 0.5
    std = 0.5
    img = img * std + mean
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.imshow(img)
    if title:
        plt.title(title)

class Generator(nn.Module):
    """ a model generates fake images """

    def __init__(self, z_dim, img_channels, features_g):
        super(Generator, self).__init__()

        # Input: N x z_dim x 1 x 1
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 32, 4, 2, 0),  # 4x4
            self._block(features_g * 32, features_g * 16, 4, 2, 1),  # 8x8
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 16x16
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 32x32
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 64x64
            nn.ConvTranspose2d(
                features_g * 2, img_channels, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)

device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')

# Create an instance of the generator
generator = Generator (z_dim = 100, img_channels=3, features_g = 32).to(device)

# Load the trained model weights
model_path = 'G.pth'
generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Use 'cuda' if you trained on GPU

# Set the model to evaluation mode
generator.eval()

# with torch.no_grad():  # Necessary for PyTorch, other frameworks may not require this
#     fixed_noise = torch.randn(16, 100, 1, 1).to(device)
#     fake = generator(fixed_noise)
#     imgs = denorm(fake).cpu().detach()
#     imgs = torchvision.utils.make_grid(imgs, nrow=4)
#     imshow(imgs, 'random generated images')

fixed_noise = torch.randn(16, 100, 1, 1).to(device)

# Generate an image using the generator
with torch.no_grad():
    generated_image = generator(fixed_noise)

# Ensure the generated image is in the correct format (e.g., channel-first or channel-last)
# If necessary, reshape or transpose it to match the format your model expects

# Convert the generated image tensor to a NumPy array and move it to CPU if needed
generated_images = generated_image.squeeze().cpu().numpy()

# Reshape the tensor to (16, 128, 128, 3) for matplotlib to display correctly
generated_images = generated_images.transpose(0, 2, 3, 1)

# Create a grid to display the images in a 4x4 layout (you can adjust this)
num_rows = 4
num_cols = 4

# Create a new figure
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

for i in range(num_rows):
    for j in range(num_cols):
        index = i * num_cols + j
        if index < len(generated_images):
            axes[i, j].imshow(generated_images[index])
            axes[i, j].axis('off')

# Adjust spacing and display the plot
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()