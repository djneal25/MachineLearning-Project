import PIL
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# data_dir = 'data2'
image_exts = ['jpeg', 'jpg', 'png']


import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import os

img_channels = 3
# from DCGAN pape
z_dim = 100
batch_size = 128

def resize_image(image, target_size=(128, 128)):
    """
    Resize an image to the target size while maintaining aspect ratio.
    The image will be scaled so that the smaller dimension fits within the target size,
    and then the image will be center cropped to the target size.

    :param image: PIL Image object.
    :param target_size: A tuple (width, height) for the target size.
    :return: Resized PIL Image object.
    """
    image.thumbnail(target_size, Image.Resampling.LANCZOS)

    # Center crop the image to the target size
    return ImageOps.fit(image, target_size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))

# class CustomImageDataset(Dataset):
#     def __init__(self, main_dir, transform=None):
#         self.main_dir = main_dir
#         self.transform = transform
#         self.total_imgs = []
#         self.labels = []
#
#
#
#         # Loop through each subdirectory
#         for label, subdir in enumerate(sorted(os.listdir(main_dir))):
#             current_dir = os.path.join(main_dir, subdir)
#             if os.path.isdir(current_dir):
#                 # Add each image file in the subdirectory
#                 for img_file in os.listdir(current_dir):
#                     if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                         image_path = f"{current_dir}/{img_file}"
#                         try:
#                             img = cv2.imread(image_path)
#                             tip = imghdr.what(image_path)
#
#                             image = Image.open(image_path).convert("RGB")  # Convert to RGB
#                             if tip not in image_exts:
#                                 print('Image not in ext list {}'.format(image_path))
#                                 os.remove(image_path)
#                         except Image.DecompressionBombWarning:
#                             print(f"Skipped large image: {img_file}")
#                             continue
#                         except Exception as e:
#                             print('Issue with image {}'.format(img_file))
#                             # os.remove(image_path)
#                         self.total_imgs.append(os.path.join(current_dir, img_file))
#                         self.labels.append(label)  # The label is determined by the subdirectory
#
#     def __len__(self):
#         return len(self.total_imgs)
#
#     def __getitem__(self, idx):
#         img_loc = self.total_imgs[idx]
#         image = Image.open(img_loc).convert("RGB")  # Convert to RGB
#         image = resize_image(image)
#
#         if self.transform is not None:
#             image = self.transform(image)
#         label = self.labels[idx]
#         return image, label


class lungDataset(Dataset):
    """Create Unsupervised Dataset from sample NIH"""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_dir + '/' + self.images[index]
        image = PIL.Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)

# Usage
tfms = transforms.Compose ([
    transforms.Resize ((128, 128)),
    transforms.ToTensor (),
    transforms.Normalize (
        [0.5 for _ in range (img_channels)], [0.5 for _ in range (img_channels)]
    )
])

# Initialize the dataset
data_folder = 'data2'  # Replace with your top directory
img_dir = 'data2/finance'
dataset = lungDataset(img_dir, transform=tfms)

# Create a DataLoader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)



# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


img_channels = 3
# from DCGAN pape
z_dim = 100
batch_size = 128
# img_size = (64, 64)
# features_d = 64
# features_g = 64


# # Visualize random batch of dataset
# img = next(iter(data_loader))
# print(img)
# img = torchvision.utils.make_grid(img, nrow=16)
# imshow(img, 'Random Batch of Images')
torch.Size([128, 3, 128, 128])


class Discriminator(nn.Module):
    """ a model that judges between real and fake images """

    def __init__(self, img_channels, features_d):
        super(Discriminator, self).__init__()

        # Input: N x channels_img x 128 x 128
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, features_d, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.LeakyReLU(0.2),
            self._block(features_d * 1, features_d * 2, 4, 2, 1),  # 32x32
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 16x16
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 8x8
            self._block(features_d * 8, features_d * 16, 4, 2, 1),  # 4x4
            nn.Conv2d(features_d * 16, 1, kernel_size=4, stride=2, padding=0),  # 1x1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


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

def initialize_weights (model):
    for m in model.modules ():
        if isinstance (m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_ (m.weight.data, 0.0, 0.02)


def test():
    """ test function to test models """
    N, in_channels, H, W = 8, 1, 128, 128
    x = torch.randn((N, in_channels, H, W))
    z_dim = 100
    D = Discriminator(in_channels, 32)
    initialize_weights(D)
    assert D(x).shape == (N, 1, 1, 1)
    G = Generator(z_dim, in_channels, 32)
    initialize_weights(G)
    z = torch.randn((N, z_dim, 1, 1))
    assert G(z).shape == (N, in_channels, H, W)
    print('Success')


test()

# Hyper Function and Initializiation
device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')

disc = Discriminator (img_channels=img_channels, features_d = 32).to(device)
gen = Generator (z_dim = 100, img_channels=img_channels, features_g = 32).to(device)

initialize_weights (disc)
initialize_weights (gen)

criterion = nn.BCELoss ()
disc_optimizer = torch.optim.Adam (disc.parameters (), lr = 0.0002, betas = (0.5, 0.999))
gen_optimizer = torch.optim.Adam (gen.parameters (), lr = 0.0002, betas = (0.5, 0.999))

def denorm (x):
    out = x * 0.5 + 0.5
    return out.clamp (0, 1)


def train_discriminator(disc, images, criterion):
    disc.train()

    batch_size = images.shape[0]
    # Loss for real images
    images = images.to(device)
    real_score = disc(images).reshape(-1)
    disc_real_loss = criterion(real_score, torch.ones_like(real_score))

    # Loss for fake images
    z = torch.randn(batch_size, z_dim, 1, 1).to(device)
    fake_images = gen(z)
    fake_score = disc(fake_images).reshape(-1)
    disc_fake_loss = criterion(fake_score, torch.zeros_like(fake_score))

    # Combine losses
    disc_loss = (disc_real_loss + disc_fake_loss) / 2

    # Reset gradients
    disc.zero_grad()

    # Compute gradients
    disc_loss.backward(retain_graph=True)

    # Adjust the parameters using backprop
    disc_optimizer.step()

    return disc_loss, real_score, fake_score


def train_generator(gen, criterion):
    gen.train()

    # Generate fake images and calculate loss
    z = torch.randn(batch_size, z_dim, 1, 1).to(device)
    fake_images = gen(z)
    output = disc(fake_images).reshape(-1)
    gen_loss = criterion(output, torch.ones_like(output))

    # Backprop and optimize
    gen.zero_grad()
    gen_loss.backward(retain_graph=True)
    gen_optimizer.step()

    return gen_loss, fake_images


from IPython.display import Image
from torchvision.utils import save_image


def save_fake_images(index, sample_dir='samples'):
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    fixed_noise = torch.randn(25, z_dim, 1, 1).to(device)
    fake = gen(fixed_noise)

    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Svaing', fake_fname)
    save_image(denorm(fake), os.path.join(sample_dir, fake_fname), nrow=5)


# Befor training
sample_dir = 'samples'
save_fake_images (0, sample_dir)
Image (os.path.join (sample_dir, 'fake_images-0000.png'))


def fit(disc, gen, dataloader, criterion, num_epochs, save_fn=None):
    #     writer_real = SummaryWriter (f'logs/real')
    #     writer_fake = SummaryWriter (f'logs/fake')
    step = 0

    d_losses, g_losses, real_scores, fake_scores = [], [], [], []
    total_step = len(data_loader)

    for epoch in range(num_epochs):
        for i, images in enumerate(data_loader):

            # Train the descriminator and generator
            disc_loss, real_score, fake_score = train_discriminator(disc, images, criterion)
            gen_loss, fake_images = train_generator(gen, criterion)

            # Inspect the losses
            if (i + 1) % 10 == 0:
                d_losses.append(disc_loss.item())
                g_losses.append(gen_loss.item())
                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                print('Epoch [{}/{}], Step [{}/{}], disc_loss: {:.4f}, gen_loss: {:.4f}, D(x): {:.2f}, D (G(z)): {:.2f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, disc_loss.item(), gen_loss.item(),
                              real_score.mean().item(), fake_score.mean().item()))

                #                 with torch.no_grad ():
                #                     fixed_noise = torch.randn (16, z_dim, 1, 1).to(device)
                #                     fake = gen (fixed_noise)

                #                     img_grid_real = torchvision.utils.make_grid (images [:16], normalize = True)
                #                     img_grid_fake = torchvision.utils.make_grid (fake [:16], normalize = True)

                #                     writer_real.add_image ('Real', img_grid_real, global_step = step)
                #                     writer_real.add_image ('Fake', img_grid_fake, global_step = step)

                step += 1

        # Sample and save images
        if save_fn is not None:
            save_fn(epoch + 1)

    return dict(disc_loss=d_losses, gen_loss=g_losses, real_score=real_scores, fake_score=fake_scores)

# Create an instance of the generator
generator = Generator()

# Load the trained model weights
model_path = 'G.pth'
generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Use 'cuda' if you trained on GPU

# Set the model to evaluation mode
generator.eval()

# fixed_noise = torch.randn (16, 100, 1, 1).to(device)
# fake = gen (fixed_noise)
# imgs = denorm (fake).cpu ().detach ()
# imgs = torchvision.utils.make_grid (imgs, nrow = 4)
# imshow (imgs, 'random generated images')

# history = fit (disc, gen, data_loader, criterion, 200, save_fake_images)

# Save the model checkpoints
# torch.save (gen.state_dict (), 'G.pth')
# torch.save (disc.state_dict (), 'D.pth')

# # create a video that show training results
# from IPython.display import FileLink
#
# files = [os.path.join (sample_dir, f) for f in os.listdir (sample_dir) if 'fake_images' in f]
# vid_fname = 'FinanceDesign,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,' \
#             '\ _dcgan_training.avi'
#
# out = cv2.VideoWriter (vid_fname, cv2.VideoWriter_fourcc(*'FMP4'), 8, (652, 652))
# [out.write (cv2.imread (fname)) for fname in files]
# out.release ()
# FileLink (vid_fname)
#
# plt.plot (history ['disc_loss'], '-')
# plt.plot (history ['gen_loss'], '-')
# plt.xlabel ('epoch')
# plt.ylabel ('loss')
# plt.legend (['Discriminator', 'Generator'])
# plt.title ('Losses')
#
# plt.plot (history ['real_score'], '-')
# plt.plot (history ['fake_score'], '-')
# plt.xlabel ('epoch')
# plt.ylabel ('score')
# plt.legend (['Real Score', 'Fake Score'])
# plt.title ('Scores')
#
# fixed_noise = torch.randn (16, 100, 1, 1).to(device)
# fake = gen (fixed_noise)
# imgs = denorm (fake).cpu ().detach ()
# imgs = torchvision.utils.make_grid (imgs, nrow = 4)
# imshow (imgs, 'random generated images')

# # Create the generator and discriminator
# netG = Generator().to(device)
# netD = Discriminator().to(device)
#
# # For content loss, Mean Squared Error (MSE) is commonly used
# criterion_content = nn.MSELoss()
#
# # For adversarial loss, Binary Cross Entropy (BCE) is typically used
# criterion_adversarial = nn.BCELoss()
#
# # Optimizers - Adam is a common choice
# optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
# optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
#
# num_epochs = 5
# real_label = 1
# fake_label = 0
#
# for epoch in range(num_epochs):
#     for i, (hr_images, _) in enumerate(data_loader):
#         # Generate low-resolution images by downsampling
#         lr_images = torch.nn.functional.interpolate(hr_images, scale_factor=0.5, mode='bicubic', align_corners=False)
#
#         # Move images to the configured device
#         lr_images = lr_images.to(device)
#         hr_images = hr_images.to(device)
#
#         # ------------------
#         #  Train Discriminator
#         # ------------------
#         netD.zero_grad()
#
#         # Real images
#         real_output = netD(hr_images)
#         real_label = torch.full((hr_images.size(0),), 1, dtype=torch.float, device=device)
#         loss_D_real = criterion_adversarial(real_output, real_label)
#
#         # Fake images
#         fake_hr = netG(lr_images).detach()  # Detach generator
#         fake_output = netD(fake_hr)
#         fake_label = torch.full((hr_images.size(0),), 0, dtype=torch.float, device=device)
#         loss_D_fake = criterion_adversarial(fake_output, fake_label)
#
#         # Total discriminator loss
#         loss_D = (loss_D_real + loss_D_fake) / 2
#         loss_D.backward()
#         optimizerD.step()
#
#         # ------------------
#         #  Train Generator
#         # ------------------
#         netG.zero_grad()
#
#         # Generate high-resolution images from low-resolution inputs
#         generated_hr = netG(lr_images)
#
#         # Content loss
#         content_loss = criterion_content(generated_hr, hr_images)
#
#         # Adversarial loss (how well the generator can fool the discriminator)
#         output = netD(generated_hr)
#         adversarial_loss = criterion_adversarial(output, real_label)
#
#         # Total generator loss
#         loss_G = content_loss + 0.001 * adversarial_loss  # The factor for adversarial loss can be tuned
#         loss_G.backward()
#         optimizerG.step()
#
#         # Print/log the training stats
#         if i % 50 == 0:  # Adjust print frequency as needed
#             print(f"[{epoch+1}/{num_epochs}][{i}/{len(data_loader)}] "
#                   f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}, "
#                   f"D(x): {real_output.mean().item():.4f}, D(G(z)): {fake_output.mean().item():.4f}")
#
# # for epoch in range(num_epochs):
# #     for i, (high_res, _) in enumerate(data_loader):
# #         # Generate low-resolution images by downsampling
# #         low_res = torch.nn.functional.interpolate(high_res, scale_factor=0.5, mode='bicubic', align_corners=False)
# #
# #         # Move images to the configured device
# #         low_res = low_res.to(device)
# #         high_res = high_res.to(device)
# #
# #         # ------------------
# #         #  Train Discriminator
# #         # ------------------
# #         # ... (discriminator training steps, similar to standard GAN training)
# #
# #         # ------------------
# #         #  Train Generator
# #         # ------------------
# #         netG.zero_grad()
# #
# #         # Generate high-resolution images from low-resolution inputs
# #         fake_high_res = netG(low_res)
# #
# #         # Calculate content loss
# #         content_loss = criterion_content(fake_high_res, high_res)
# #
# #         # Calculate adversarial loss
# #         fake_output = netD(fake_high_res)
# #         adversarial_loss = criterion_adversarial(fake_output, torch.ones_like(fake_output, device=device))
# #
# #         # Total generator loss
# #         generator_loss = content_loss + 0.001 * adversarial_loss
# #         generator_loss.backward()
# #         optimizerG.step()
# #
# #         # Print the training progress
# #         if i % 50 == 0:
# #             print(f"[{epoch+1}/{num_epochs}][{i}/{len(data_loader)}] "
# #                   f"Loss_D: {loss_D.item():.4f}, Loss_G: {generator_loss.item():.4f}")
#
#
#
# # import torch
# # import torch.nn as nn
# #
# # class Generator(nn.Module):
# #     def __init__(self):
# #         super(Generator, self).__init__()
# #         self.initial = nn.Sequential(
# #             nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
# #             nn.PReLU()
# #         )
# #         self.residuals = nn.Sequential(*[ResidualBlock() for _ in range(5)])
# #         self.upsample = nn.Sequential(
# #             nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
# #             nn.PixelShuffle(2),
# #             nn.PReLU()
# #         )
# #         self.output = nn.Sequential(
# #             nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),
# #             nn.Tanh()
# #         )
# #
# #     def forward(self, x):
# #         x = self.initial(x)
# #         residual = x.clone()
# #         for block in self.residuals:
# #             x = block(x)
# #         x += residual
# #         x = self.upsample(x)
# #         x = self.output(x)
# #         return x
# #
# # class ResidualBlock(nn.Module):
# #     def __init__(self):
# #         super(ResidualBlock, self).__init__()
# #         self.block = nn.Sequential(
# #             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
# #             nn.BatchNorm2d(64),
# #             nn.PReLU(),
# #             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
# #             nn.BatchNorm2d(64)
# #         )
# #
# #     def forward(self, x):
# #         return x + self.block(x)
# #
# #
# # class Discriminator(nn.Module):
# #     def __init__(self):
# #         super(Discriminator, self).__init__()
# #         self.model = nn.Sequential(
# #             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
# #             nn.LeakyReLU(0.2, inplace=True),
# #
# #             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
# #             nn.BatchNorm2d(64),
# #             nn.LeakyReLU(0.2, inplace=True),
# #
# #             # Additional layers...
# #
# #             nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
# #             nn.BatchNorm2d(512),
# #             nn.LeakyReLU(0.2, inplace=True),
# #
# #             nn.AdaptiveAvgPool2d(1),
# #             nn.Conv2d(512, 1024, kernel_size=1),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             nn.Conv2d(1024, 1, kernel_size=1),
# #             nn.Sigmoid()
# #         )
# #
# #     def forward(self, x):
# #         return self.model(x)
# #
# # criterion_content = nn.MSELoss()
# # criterion_GAN = nn.BCEWithLogitsLoss()
# #
# # # Initialize networks
# # generator = Generator().to(device)
# # discriminator = Discriminator().to(device)
# #
# # # Optimizers
# # optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# #
# # num_epochs = 5  # Adjust as needed
# # print_frequency = 100  # How often to print the training status
# #
# # for epoch in range(num_epochs):
# #     for i, (hr_images, labels) in enumerate(data_loader):
# #         # Generate low-resolution images by downsampling
# #         # Adjust scale_factor as needed for your specific use case
# #         lr_images = torch.nn.functional.interpolate(hr_images, scale_factor=0.5, mode='bicubic', align_corners=False)
# #
# #         # Move images to the configured device
# #         lr_images = lr_images.to(device)
# #         hr_images = hr_images.to(device)
# #
# #         # ------------------
# #         #  Train Discriminator
# #         # ------------------
# #         optimizer_D.zero_grad()
# #
# #         # Real images
# #         real_output = discriminator(hr_images)
# #         real_label = torch.ones_like(real_output, device=device)
# #         loss_real = criterion_GAN(real_output, real_label)
# #
# #         # Fake images
# #         generated_hr = generator(lr_images)
# #         fake_output = discriminator(generated_hr.detach())
# #         fake_label = torch.zeros_like(fake_output, device=device)
# #         loss_fake = criterion_GAN(fake_output, fake_label)
# #
# #         # Total discriminator loss
# #         d_loss = (loss_real + loss_fake) / 2
# #         d_loss.backward()
# #         optimizer_D.step()
# #
# #         # ------------------
# #         #  Train Generator
# #         # ------------------
# #         optimizer_G.zero_grad()
# #
# #         # Adversarial loss (how well the generator can fool the discriminator)
# #         fake_output = discriminator(generated_hr)
# #         gen_label = torch.ones_like(fake_output, device=device)
# #         adversarial_loss = criterion_GAN(fake_output, gen_label)
# #
# #         # Content loss (how similar the generated image is to the original)
# #         content_loss = criterion_content(generated_hr, hr_images)
# #
# #         # Total generator loss
# #         g_loss = content_loss + 0.001 * adversarial_loss
# #         g_loss.backward()
# #         optimizer_G.step()
# #
# #         # Print/log the losses
# #         if (i + 1) % print_frequency == 0:
# #             print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], "
# #                   f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
#
# # Note: labels are not used in this SRGAN training loop.
#
# # import torchvision.transforms as transforms
# # import matplotlib.pyplot as plt
# #
# # def show_image(img_tensor):
# #     img = img_tensor.detach().cpu()
# #     img = transforms.ToPILImage()(img)
# #     plt.imshow(img)
# #     plt.axis('off')
# #
# # num_epochs = 5  # Or however many you choose
# # for epoch in range(num_epochs):
# #     for i, (images, _) in enumerate(data_loader):
# #         # [Previous training steps]
# #         # Prepare real and fake labels
# #         real_labels = torch.ones(images.size(0), 1).to(device)
# #         fake_labels = torch.zeros(images.size(0), 1).to(device)
# #
# #         # Move to device
# #         high_res_real = images.to(device)
# #         low_res = torch.nn.functional.interpolate(high_res_real, scale_factor=0.5)
# #
# #         # Train Discriminator
# #         discriminator.zero_grad()
# #         outputs = discriminator(high_res_real)
# #         d_loss_real = criterion_GAN(outputs, real_labels)
# #         d_loss_real.backward()
# #
# #         high_res_fake = generator(low_res)
# #         outputs = discriminator(high_res_fake.detach())
# #         d_loss_fake = criterion_GAN(outputs, fake_labels)
# #         d_loss_fake.backward()
# #
# #         d_loss = d_loss_real + d_loss_fake
# #         optimizer_D.step()
# #
# #         # Train Generator
# #         generator.zero_grad()
# #         outputs = discriminator(high_res_fake)
# #         g_loss = criterion_GAN(outputs, real_labels) + 0.001 * criterion_content(high_res_fake, high_res_real)
# #         g_loss.backward()
# #         optimizer_G.step()
# #         # Logging
# #         if (i + 1) % 100 == 0:  # Log every 100 steps
# #             print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], '
# #                   f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
# #
# #         # Visualization
# #         if (i + 1) % 500 == 0:  # Visualize every 500 steps
# #             with torch.no_grad():
# #                 # Take the first image in the batch for visualization
# #                 high_res_fake_sample = generator(low_res[:1])
# #                 plt.figure(figsize=(10, 4))
# #                 plt.subplot(1, 3, 1)
# #                 show_image(high_res_real[0])
# #                 plt.title('Original High-Res')
# #                 plt.subplot(1, 3, 2)
# #                 show_image(low_res[0])
# #                 plt.title('Low-Res')
# #                 plt.subplot(1, 3, 3)
# #                 show_image(high_res_fake_sample[0])
# #                 plt.title('Generated High-Res')
# #                 plt.show()
# #
# #
