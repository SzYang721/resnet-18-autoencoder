import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        _, features = self.encoder(x)
        decoded = self.decoder(features)
        return decoded

    def set_requires_grad(self, model, value):
        for param in model.parameters():
            param.requires_grad = value

    def freeze_encoder(self):
        self.set_requires_grad(self.encoder, False)

    def freeze_decoder(self):
        self.set_requires_grad(self.decoder, False)

    def unfreeze_encoder(self):
        self.set_requires_grad(self.encoder, True)

    def unfreeze_decoder(self):
        self.set_requires_grad(self.decoder, True)