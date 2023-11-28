# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        ## vit_h = 1280, vit_l = 1024
        vit_dim: int = 1280
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
        
        # HQ-SAM parameters
        self.hf_token = nn.Embedding(1, transformer_dim) # HQ-Ouptput-Token
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) # corresponding new MLP layer for HQ-Ouptput-Token
        self.num_mask_tokens = self.num_mask_tokens + 1
        
        # three conv fusion layers for obtaining HQ-Feature
        self.compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU(), 
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )
        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))
        

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        interm_embeddings: torch.Tensor,
        prompt_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            hq_features=hq_features,
            prompt_img=prompt_img
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks_sam = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens)]
        masks = masks_sam + masks_hq

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_features: torch.Tensor,
        prompt_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features.repeat(b,1,1,1)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:self.num_mask_tokens-1] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_sam_hq = (hyper_in[:,self.num_mask_tokens-1:] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_sam_hq],dim=1)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
    
    
        
#     self.patch_backbone = Backbone('resnet50',
#                 train_backbone=True,
#                 return_layer='layer3',
#                 frozen_bn=False,
#                 dilation=False)
#     self.EPF_extractor = DirectPooling(input_dim=1024, hidden_dim=256)  # pooling used for the query patch feature
    
#     self.matcher = InnerProductMatcher()
#     prompt_img_patch_feautre = self.patch_backbone(prompt_img_patch)
#     tmp_patch = self.EPF_extractor(prompt_img_patch_feautre) # compress the feature maps into vectors and inject scale embeddings
    
#     ##                                      256 64 64, num_img 1 256
#     _, corr_map = self.matcher(image_features, tmp_patch)
    
    
    
# class BackboneBase(nn.Module):
#     def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_layer: str):
#         super().__init__()
#         for name, parameter in backbone.named_parameters():
#             if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
#             #if not train_backbone:
#                 parameter.requires_grad_(False)
        
#         return_layers = {return_layer: '0'}
#         self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
#         self.num_channels = num_channels

#     def forward(self, tensor_list):
#         """supports both NestedTensor and torch.Tensor
#         """
#         out = self.body(tensor_list)
#         return out['0']

# class Backbone(BackboneBase):
#     """ResNet backbone with frozen BatchNorm."""

#     def __init__(self, name: str,
#                  train_backbone: bool,
#                  return_layer: str,
#                  frozen_bn: bool,
#                  dilation: bool):
        
#         if frozen_bn:
#             backbone = getattr(torchvision.models, name)(
#                                replace_stride_with_dilation=[False, False, dilation],
#                                pretrained=True, norm_layer=FrozenBatchNorm2d)
#         else:
#             backbone = getattr(torchvision.models, name)(
#                                replace_stride_with_dilation=[False, False, dilation],
#                                pretrained=True)
            
#         # load the SwAV pre-training model from the url instead of supervised pre-training model
#         if name == 'resnet50':
#             #checkpoint = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',map_location="cpu")
#             checkpoint = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',map_location="cpu")
#             #checkpoint = torch.load('nvidia_resnet50_200821.pth.tar')
#             state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
#             backbone.load_state_dict(state_dict, strict=False)
#             #pass
#         if name in ('resnet18', 'resnet34'):
#             num_channels = 512
#         else:
#             if return_layer == 'layer3':
#                 num_channels = 1024
#             else:
#                 num_channels = 2048
#         super().__init__(backbone, train_backbone, num_channels, return_layer)


# """
# Class agnostic counting
# Feature extractors for exemplars.
# """
# from torch import nn
# import pdb
# class DirectPooling(nn.Module):
#     def __init__(self, input_dim, hidden_dim, repeat_times=1, use_scale_embedding=True, scale_number=20):
#         super().__init__()
#         self.repeat_times = repeat_times
#         self.use_scale_embedding = use_scale_embedding
#         self.patch2query = nn.Linear(input_dim, hidden_dim) # align the patch feature dim to query patch dim.
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # pooling used for the query patch feature
#         self._weight_init_()
#         # if self.use_scale_embedding:
#         #     self.scale_embedding = nn.Embedding(scale_number, hidden_dim)
    
#     def forward(self, patch_feature):
#         batch_num_patches = patch_feature.shape[0]
#         patch_feature = self.avgpool(patch_feature).flatten(1) # bs X patchnumber X feature_dim
#         #patch_feature = patch_feature \
#         patch_feature = self.patch2query(patch_feature) \
#             .view(1, batch_num_patches, -1) \
#             .repeat_interleave(self.repeat_times, dim=1) \
#             .permute(1, 0, 2) \
#             .contiguous() 
#         # if self.use_scale_embedding:
#         #     scale_embedding = self.scale_embedding(scale_index) # bs X number_query X dim
#         #     patch_feature = patch_feature + scale_embedding.permute(1, 0, 2)
        
#         return patch_feature
    
#     def _weight_init_(self):
#         for p in self.parameters():
#                 if p.dim() > 1:
#                     nn.init.xavier_uniform_(p)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.01)
#                 # nn.init.kaiming_uniform_(
#                 #         m.weight, 
#                 #         mode='fan_in', 
#                 #         nonlinearity='relu'
#                 #         )
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

# class DynamicSimilarityMatcher(nn.Module):
#     def __init__(self, hidden_dim, proj_dim, dynamic_proj_dim, activation='tanh', pool='mean', use_bias=False):
#         super().__init__()
#         self.query_conv = nn.Linear(in_features=hidden_dim, out_features=proj_dim, bias=use_bias)
#         self.key_conv = nn.Linear(in_features=hidden_dim, out_features=proj_dim, bias=use_bias)
#         self.dynamic_pattern_conv = nn.Sequential(nn.Linear(in_features=proj_dim, out_features=dynamic_proj_dim),
#                                           nn.ReLU(),
#                                           nn.Linear(in_features=dynamic_proj_dim, out_features=proj_dim))
        
#         self.softmax  = nn.Softmax(dim=-1)
#         self._weight_init_()
        
#         if activation == 'relu':
#             self.activation = nn.ReLU()
#         elif activation == 'leaky_relu':
#             self.activation = nn.LeakyReLU()
#         elif activation == 'sigmoid':
#             self.activation = nn.Sigmoid()
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()
#         elif activation == 'none':
#             self.activation = None
#         else:
#             raise NotImplementedError
            
#     def forward(self, features, patches):
#         bs, c, h, w = features.shape
#         features = features.flatten(2).permute(2, 0, 1)  # hw * bs * dim
        
#         proj_feat = self.query_conv(features)
#         patches_feat = self.key_conv(patches)
#         patches_ca = self.activation(self.dynamic_pattern_conv(patches_feat))
        
#         proj_feat = proj_feat.permute(1, 0, 2)
#         patches_feat = (patches_feat * (patches_ca + 1)).permute(1, 2, 0)  # bs * c * exemplar_number        
#         energy = torch.bmm(proj_feat, patches_feat)                        # bs * hw * exemplar_number

#         corr = energy.mean(dim=-1, keepdim=True)
#         out = features.permute(1,0,2)  # hw * bs * c
#         out = torch.cat((out, corr), dim=-1)
        
#         out = out.permute(1,0,2)
#         return out.permute(1, 2, 0).view(bs, c+1, h, w), energy 
    
#     def _weight_init_(self):
#         for p in self.parameters():
#                 if p.dim() > 1:
#                     nn.init.xavier_uniform_(p)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.01)
#                 # nn.init.kaiming_uniform_(
#                 #         m.weight, 
#                 #         mode='fan_in', 
#                 #         nonlinearity='relu'
#                 #         )
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

# class InnerProductMatcher(nn.Module):
#     def __init__(self, pool='mean'):
#         super().__init__()
#         self.pool = pool
    
#     def forward(self, features, patches_feat):
#         # bs, c, h, w = features.shape
#         ##                                      256 64 64, num_img 1 256
        
#         c, h, w = features.shape
#         # 4096, 256
#         features = features.flatten(1).permute(1, 0).unsqueeze(0)
#         # 256, exeplar_num 
#         patches_feat = patches_feat.permute(1, 2, 0)      # bs * c * exemplar_number
#         energy = torch.bmm(features, patches_feat)       # bs * hw * exemplar_number
#         if self.pool == 'mean':
#             corr = energy.mean(dim=-1, keepdim=True)
#         elif self.pool == 'max':
#             corr = energy.max(dim=-1, keepdim=True)[0]
#         out = torch.cat((features, corr), dim=-1) # bs * hw * dim
#         return out.permute(0, 2, 1).view(1, c+1, h, w), energy