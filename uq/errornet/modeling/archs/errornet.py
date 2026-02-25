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


class LayerNorm(torch.nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x