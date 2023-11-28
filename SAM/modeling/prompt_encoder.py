# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d

from torch.nn import Dropout
from operator import mul
import math
from functools import reduce

from .transformer import TwoWayAttentionBlock
import time
import os
import matplotlib.pyplot as plt

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

import numpy as np
import torch
from torch import nn
from typing import Any, Optional, Tuple, Type
from .common import LayerNorm2d


# import torchvision.models.resnet

class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        
        ### Box prompt modulated
        # mlp_dim = 2048
        # num_heads= 8
        # attention_downsample_rate= 2
        # self.box_prompt_modulate = TwoWayAttentionBlock(
        #     embedding_dim=embed_dim,
        #     num_heads=num_heads,
        #     mlp_dim=mlp_dim,
        #     activation=activation,
        #     attention_downsample_rate=attention_downsample_rate,
        #     skip_first_layer_pe=False,
        #     # skip_first_layer_pe=True,
        # )
        
        
        # self.patch_backbone = Backbone('resnet50',
        #             train_backbone=True,
        #             return_layer='layer3',
        #             frozen_bn=False,
        #             dilation=False)
        # self.EPF_extractor = DirectPooling(input_dim=1024, hidden_dim=256)  # pooling used for the query patch feature
        
        # self.matcher = InnerProductMatcher()
        self.no_mask_embed = nn.Embedding(1, embed_dim)
        # self.no_imgs_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
                
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _embed_imags(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs mask inputs."""
        imgs_embedding = self.imgs_downscaling(imgs)
        return imgs_embedding
    
    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        # if imgs is not None:
        #     imgs_dense_embeddings = self._embed_imags(imgs)
        # else:
        #     imgs_dense_embeddings = self.no_imgs_embed.weight.reshape(1, -1, 1, 1).expand(
        #         bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            # )
            
        return sparse_embeddings, dense_embeddings

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_layer: str):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            #if not train_backbone:
                parameter.requires_grad_(False)
        
        return_layers = {return_layer: '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list):
        """supports both NestedTensor and torch.Tensor
        """
        out = self.body(tensor_list)
        return out['0']

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_layer: str,
                 frozen_bn: bool,
                 dilation: bool):
        
        if frozen_bn:
            backbone = getattr(torchvision.models, name)(
                               replace_stride_with_dilation=[False, False, dilation],
                               pretrained=True, norm_layer=FrozenBatchNorm2d)
        else:
            backbone = getattr(torchvision.models, name)(
                               replace_stride_with_dilation=[False, False, dilation],
                               pretrained=True)
            
        # load the SwAV pre-training model from the url instead of supervised pre-training model
        if name == 'resnet50':
            #checkpoint = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',map_location="cpu")
            checkpoint = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',map_location="cpu")
            #checkpoint = torch.load('nvidia_resnet50_200821.pth.tar')
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            backbone.load_state_dict(state_dict, strict=False)
            #pass
        if name in ('resnet18', 'resnet34'):
            num_channels = 512
        else:
            if return_layer == 'layer3':
                num_channels = 1024
            else:
                num_channels = 2048
        super().__init__(backbone, train_backbone, num_channels, return_layer)


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    
def save_plt(image, path=None):
    if type(image) is torch.Tensor:
        image = image.detach().cpu().numpy()
    save_base = os.path.join('./feature_results')
    plt.figure(figsize=(10,10))
    plt.imshow(image, cmap='gray')
    filename = f"{path}.png"
    plt.savefig(os.path.join(save_base, filename), bbox_inches='tight', pad_inches=0)
    plt.close()

"""
Class agnostic counting
Feature extractors for exemplars.
"""
from torch import nn
import pdb
class DirectPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, repeat_times=1, use_scale_embedding=True, scale_number=20):
        super().__init__()
        self.repeat_times = repeat_times
        self.use_scale_embedding = use_scale_embedding
        self.patch2query = nn.Linear(input_dim, hidden_dim) # align the patch feature dim to query patch dim.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # pooling used for the query patch feature
        self._weight_init_()
        # if self.use_scale_embedding:
        #     self.scale_embedding = nn.Embedding(scale_number, hidden_dim)
    
    def forward(self, patch_feature):
        batch_num_patches = patch_feature.shape[0]
        patch_feature = self.avgpool(patch_feature).flatten(1) # bs X patchnumber X feature_dim
        #patch_feature = patch_feature \
        patch_feature = self.patch2query(patch_feature) \
            .view(1, batch_num_patches, -1) \
            .repeat_interleave(self.repeat_times, dim=1) \
            .permute(1, 0, 2) \
            .contiguous() 
        # if self.use_scale_embedding:
        #     scale_embedding = self.scale_embedding(scale_index) # bs X number_query X dim
        #     patch_feature = patch_feature + scale_embedding.permute(1, 0, 2)
        
        return patch_feature
    
    def _weight_init_(self):
        for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight, 
                #         mode='fan_in', 
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DynamicSimilarityMatcher(nn.Module):
    def __init__(self, hidden_dim, proj_dim, dynamic_proj_dim, activation='tanh', pool='mean', use_bias=False):
        super().__init__()
        self.query_conv = nn.Linear(in_features=hidden_dim, out_features=proj_dim, bias=use_bias)
        self.key_conv = nn.Linear(in_features=hidden_dim, out_features=proj_dim, bias=use_bias)
        self.dynamic_pattern_conv = nn.Sequential(nn.Linear(in_features=proj_dim, out_features=dynamic_proj_dim),
                                          nn.ReLU(),
                                          nn.Linear(in_features=dynamic_proj_dim, out_features=proj_dim))
        
        self.softmax  = nn.Softmax(dim=-1)
        self._weight_init_()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise NotImplementedError
            
    def forward(self, features, patches):
        bs, c, h, w = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # hw * bs * dim
        
        proj_feat = self.query_conv(features)
        patches_feat = self.key_conv(patches)
        patches_ca = self.activation(self.dynamic_pattern_conv(patches_feat))
        
        proj_feat = proj_feat.permute(1, 0, 2)
        patches_feat = (patches_feat * (patches_ca + 1)).permute(1, 2, 0)  # bs * c * exemplar_number        
        energy = torch.bmm(proj_feat, patches_feat)                        # bs * hw * exemplar_number

        corr = energy.mean(dim=-1, keepdim=True)
        out = features.permute(1,0,2)  # hw * bs * c
        out = torch.cat((out, corr), dim=-1)
        
        out = out.permute(1,0,2)
        return out.permute(1, 2, 0).view(bs, c+1, h, w), energy 
    
    def _weight_init_(self):
        for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight, 
                #         mode='fan_in', 
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class InnerProductMatcher(nn.Module):
    def __init__(self, pool='mean'):
        super().__init__()
        self.pool = pool
    
    def forward(self, features, patches_feat):
        # bs, c, h, w = features.shape
        ##                                      256 64 64, num_img 1 256
        
        c, h, w = features.shape
        # 4096, 256
        features = features.flatten(1).permute(1, 0).unsqueeze(0)
        # 256, exeplar_num 
        patches_feat = patches_feat.permute(1, 2, 0)      # bs * c * exemplar_number
        energy = torch.bmm(features, patches_feat)       # bs * hw * exemplar_number
        if self.pool == 'mean':
            corr = energy.mean(dim=-1, keepdim=True)
        elif self.pool == 'max':
            corr = energy.max(dim=-1, keepdim=True)[0]
        out = torch.cat((features, corr), dim=-1) # bs * hw * dim
        return out.permute(0, 2, 1).view(1, c+1, h, w), energy