import torch

class ErrorNet(torch.nn.Module):
    def __init__(self, embed_dim=60, depths=[60, 60, 60], upscale=4, dropout=True, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.depths = [embed_dim] + depths
        self.upscale = upscale
        self.dropout = dropout

        # Feature extraction bottlenech.
        modules = []
        modules.append(torch.nn.BatchNorm2d(self.depths[0])) # pre-norm
        for i in range(len(depths)):
            modules.append(torch.nn.Conv2d(self.depths[i], self.depths[i + 1], kernel_size=3, stride=1, padding='same'))
            modules.append(torch.nn.ReLU())
            modules.append(torch.nn.BatchNorm2d(self.depths[i+1]))
            if self.dropout:
                modules.append(torch.nn.Dropout(p=0.25))
        self.bottleneck = torch.nn.Sequential(*modules)

        # Output layer.
        self.out = torch.nn.Conv2d(depths[-1], 1, kernel_size=1, stride=1, padding='same')


    def forward(self, x):

        # Initial upsampling
        if self.upscale!=0:
            x = torch.nn.functional.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)

        # Bottleneck
        x = self.bottleneck(x)

        # Prediction layer
        error = self.out(x)

        return error
