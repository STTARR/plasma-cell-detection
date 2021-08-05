import torch
from torch import nn
from torchvision import models
import copy
import numpy as np


class UNetVGG(nn.Module):
    """
    U-Net-style model generated from PyTorch's pre-trained VGG encoders and a U-Net style decoder.
    
    Note all pretrained models expect 3-channel RGB images normalized to
    mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].    
    See: https://pytorch.org/docs/stable/torchvision/models.html
    """
    def __init__(self, vggname, num_classes):
        super().__init__()
        
        if vggname not in models.vgg.model_urls:
            raise ValueError(f"'vggmodel' must be one of: {list(models.vgg.model_urls.keys())}")
        self.vggname = vggname
        if "_bn" in vggname:
            self.batch_norm = True  # batch norm decoder blocks as well if used
        
        self.num_classes = num_classes
        encoder, midblock = self._init_encoder()
        
        # Generate decoder by iterating backwards through encoder blocks
        decoder = []
        for block in ([midblock] + encoder[::-1][:-1]):
            in_ch = next(c for c in block if isinstance(c, nn.Conv2d)).out_channels
            out_ch = next(c for c in block if isinstance(c, nn.Conv2d)).in_channels
            up = [
                # Upsampling
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
            ]
            conv = [
                # Convolutions
                # First layer takes double the number of feature channels due to skip connection
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
                nn.ReLU()
            ]
            if self.batch_norm:
                conv.append(nn.BatchNorm2d(out_ch))
            conv.extend([
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU()
            ])
            if self.batch_norm:
                conv.append(nn.BatchNorm2d(out_ch))
            
            # add blocks to decoder
            decoder.append(nn.Sequential(
               nn.Sequential(*up), nn.Sequential(*conv)
            ))
        
        self.encoder = nn.ModuleList(encoder)
        self.midblock = midblock
        self.decoder = nn.ModuleList(decoder)
        self.classify = nn.Conv2d(out_ch, num_classes, kernel_size=1)
    
    def _init_encoder(self):
        """Initialize encoder and middle block of U-Net-style architecture with vggmodel."""
        vggmodel = getattr(models.vgg, self.vggname)(pretrained=True)
        features = vggmodel.features
        
        encoders = []
        # Iterate through convolutional layers and split by identifying MaxPool2d layers
        # (Necessary since we need to pass the encoder activations at each block to the decoder)
        curr_block = []
        for layer in features:
            if isinstance(layer, nn.MaxPool2d):
                encoders.append(nn.Sequential(*curr_block))
                curr_block = []
            curr_block.append(copy.deepcopy(layer))
        return encoders[:-1], encoders[-1]
    
    def freeze_encoder_blocks(self, n=None):
        """Freeze first n blocks of pre-trained weights in the encoder.
        If n is not supplied, all encoder blocks are frozen."""
        n = n if n is not None else len(self.encoder)
        for i, block in enumerate(self.encoder[:n]):
            for param in block.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        encode = []
        for encode_block in self.encoder:
            x = encode_block(x)
            encode.append(x.clone())  # Need clone - don't want to append the reference
            
        # the last ('center') block doesn't need to be kept
        x = self.midblock(x)
        
        for i, (decode_up, decode_conv) in enumerate(self.decoder):
            # Append input with encoder block output of same shape
            x = decode_up(x)
            x = decode_conv(torch.cat([encode[-(i+1)], x], dim=1))
        
        return self.classify(x)
