import segmentation_models_pytorch as smp
import torch.nn as nn

from typing import Optional, Union, List
from torch.nn import functional as F
from model.model_scripts.model_utils import UnetDecoder


class UnetCustom(smp.Unet):
  def __init__(
      self,
      encoder_name: str = "resnet50",
      encoder_depth: int = 5, # depth 5 is default setting
      encoder_weights: str = "imagenet",
      decoder_use_batchnorm: bool = True,
      decoder_channels: List[int] = (256, 128, 64, 32, 16),
      decoder_attention_type: Optional[str] = None,
      in_channels: int = 3,
      classes: int = 1,
      activation: Optional[Union[str, callable]] = None,
      aux_params: Optional[dict] = None,
    ):
      super().__init__(
        encoder_name=encoder_name,
        encoder_depth=encoder_depth,
        encoder_weights=encoder_weights,
        decoder_use_batchnorm=decoder_use_batchnorm,
        decoder_channels=decoder_channels,
        decoder_attention_type=decoder_attention_type,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
        aux_params=aux_params,
      )
        
      self.segmentation_head = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=classes, kernel_size=1, padding=0, bias=False)
      )
      smp.base.initialization.initialize_head(self.segmentation_head)
      self.decoder = UnetDecoder(encoder_channels=self.encoder.out_channels,
                                decoder_channels=decoder_channels)

  def forward(self, x):
    features = self.encoder(x)
    decoder_output, layerwise_features = self.decoder(*features)
    masks = self.segmentation_head(decoder_output)
    return masks, decoder_output, layerwise_features


class projection_head_1x1(nn.Module):
  def __init__(self):
    super().__init__()
    self.projection_head = nn.Sequential(
      nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(num_features=16),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(num_features=32)
    )
    smp.base.initialization.initialize_head(self.projection_head)

  def forward(self, x):
    projections = self.projection_head(x)
    return projections
    

class projection_head_4x4(nn.Module):
  def __init__(self):
    super().__init__()
    self.projection_head = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(num_features=32),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(num_features=32)
    )
    smp.base.initialization.initialize_head(self.projection_head)

  def forward(self, x):
    projections = self.projection_head(x)
    return projections