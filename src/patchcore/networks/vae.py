import torch
from torch import nn
import torch.nn.functional as F

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()

        # input 32 x 3 x 512 x 512

        self.conv1 = nn.Conv2d(3, 3, 3, stride=2, padding=1)  # 3 x 256 x 256
        self.conv2 = nn.Conv2d(3, 32, 3, stride=2, padding=1)  # 32 x 128 x 128
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 32 x 64 x 64
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 64 x 32 x 32
        self.conv5 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 64 x 16 x 16
        self.conv6 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 64 x 8 x 8
        self.conv7 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 64 x 4 x 4
        self.conv8 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # 128 x 2 x 2
        self.linear1 = nn.Linear(128 * 2 * 2, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = x.to(DEVICE)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        x = torch.flatten(x, start_dim=1)

        x = F.leaky_relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):

    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 512),
            nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, 2, 2))
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding='same'),
            nn.LeakyReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, stride=1, padding='same'),
            nn.LeakyReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, stride=1, padding='same'),
            nn.LeakyReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, stride=1, padding='same'),
            nn.LeakyReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding='same'),
            nn.LeakyReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, 3, stride=1, padding='same'),
            nn.LeakyReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding='same'),
            nn.LeakyReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 16, 3, stride=1, padding='same'),
            nn.LeakyReLU(True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(16, 3, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class VariationalAutoencoder(nn.Module):
    seed = 0

    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(DEVICE)
        z = self.encoder(x)
        return self.decoder(z)


def get_vae(latent_dims=512, pretrained=False):
    model = VariationalAutoencoder(latent_dims=latent_dims)
    if pretrained:
        model.load_state_dict(torch.load("/mnt/d/Notebooks/Advanced_DL/SynthFaceVAE1.0/weights.pt"))
    return model
