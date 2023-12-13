# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
import numpy as np
from einops import rearrange, reduce

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
        self.vit_dim = vit_dim
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

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        model_type = "vit_h"
        checkpoint_dict = {"vit_b":"pre_trained/sam_vit_b_maskdecoder.pth",
                    "vit_l":"pre_trained/sam_vit_l_maskdecoder.pth",
                    'vit_h':"pre_trained/sam_vit_h_maskdecoder.pth"}
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("HQ Decoder init from SAM MaskDecoder")
        for n,p in self.named_parameters():
            p.requires_grad = False

        ### Addition parameters start
        self.hf_token = nn.Embedding(1, transformer_dim) # HQ-Ouptput-Token
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) # corresponding new MLP layer for HQ-Ouptput-Token
        
        self.num_hq_token = 1
        self.additional_token = 4
        self.total_num_mask_tokens = self.num_mask_tokens + self.num_hq_token + self.additional_token
        
        self.output_hypernetworks_additional_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.additional_token)
            ]
        )
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
        
        self.EPF_extractor = DirectPooling(input_dim=1280, hidden_dim=256)  # pooling used for the query patch feature
        self.feature_align = nn.Linear(4096, 256) # align the patch feature dim to query patch dim.
        self.matcher = InnerProductMatcher()
        
        # embed_dim = 32
        # mid_dim = 1024
        # head= 8
        # dropout = 0.0
        # self.aggt = SimilarityWeightedAggregation(embed_dim=embed_dim, head=head, dropout=dropout)
        # self.conv1 = nn.Conv2d(embed_dim, mid_dim, kernel_size=3, stride=1, padding=1)
        # self.dropout = nn.Dropout(dropout)
        # self.conv2 = nn.Conv2d(mid_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        # self.norm1 = nn.LayerNorm(embed_dim)
        # self.norm2 = nn.LayerNorm(embed_dim)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # self.activation = nn.LeakyReLU()
        
        # safecount_block = SAFECountBlock(
        #     embed_dim=embed_dim,
        #     mid_dim=mid_dim,
        #     head=head,
        #     dropout=dropout,
        #     activation=activation,
        # )


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        interm_embeddings: torch.Tensor,
        prompt_box: torch.Tensor
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
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2).contiguous() # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        # torch.Size([1, 1280, 64, 64]
        normalized_box = np.divide(prompt_box[0], self.vit_dim // image_embeddings.size(3))
        x, y, w, h = np.around(normalized_box)
        prompt_img_feature = vit_features.clone()[:, :, int(y) : int(y+h), int(x) : int(x+w)]
        # prompt_img_feature_x2 = vit_features.clone()[:, :, int(y) : int(y+h*2), int(x) : int(x+w*2)]
        
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            hq_features=hq_features,
            prompt_img_feature=prompt_img_feature,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks_sam = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens)]

        # Adddion augmented masks
        for i in range(self.num_mask_tokens + self.num_hq_token, self.total_num_mask_tokens+1):
            masks_hq += masks[:, slice(i-1, i)]

        # print(masks_sam.shape, masks_hq.shape)
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
        prompt_img_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        # tgt = prompt_img_feature.contiguous()
        # tgt2 = self.aggt(query=tgt, keys=hq_features, values=hq_features).contiguous()
        # tgt = tgt + self.dropout1(tgt2).contiguous()
        # tgt = tgt.permute(0, 2, 3, 1).contiguous()
        # tgt = self.norm1(tgt).permute(0, 3, 1, 2).contiguous()
        # tgt2 = self.conv2(self.dropout(self.activation(self.conv1(tgt)))).contiguous()
        # tgt = tgt + self.dropout2(tgt2).contiguous()
        # tgt = tgt.permute(0, 2, 3, 1).contiguous()
        # tgt = self.norm2(tgt).permute(0, 3, 1, 2).contiguous()
        # # tgt = F.adaptive_max_pool2d(tgt, [1,1], return_indices=False)
        # tgt = reduce(tgt, 'b c h w -> b c', 'max').contiguous()
        # hq_token = self.feature_align(tgt).contiguous()
        
        # corr_map = self.matcher(image_embeddings, tmp_patch)
        
        
        augmented_prompt_imgs = [prompt_img_feature, 
                                torch.fliplr(prompt_img_feature),
                                torch.flipud(prompt_img_feature),
                                torch.rot90(prompt_img_feature, 1, [2,3])] 
        
        hq_tokens = []
        for prompt_img_feature in augmented_prompt_imgs:
            tmp_patch = self.EPF_extractor(prompt_img_feature) # compress the feature maps into vectors and inject scale embeddings
            corr_map = self.matcher(image_embeddings, tmp_patch)
            corr_map = corr_map.squeeze(-1)
            hq_token = self.feature_align(corr_map)
            hq_tokens.append(hq_token)
        hq_tokens = torch.cat(hq_tokens, dim=0) 
        
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight, hq_tokens], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        
        
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        ## add coor_map
        # src = src + corr_map
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        # hs size = 1 8 256
        iou_token_out = hs[:, 0, :]
        
        # 기존 4 + hq + 1 mask_tokens_out = 1 5 256
        mask_tokens_out = hs[:, 1 : (1 + self.total_num_mask_tokens), :]
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w).contiguous()

        # src.size() torch.Size([1, 256, 64, 64])
        upscaled_embedding_sam = self.output_upscaling(src)
        # torch.Size([1, 32, 256, 256]) torch.Size([1, 32, 256, 256])
        # print(upscaled_embedding_sam.size(), hq_features.size())
        
        # self.embedding_maskfeature = 피쳐양 줄였다가 늘리면서 hq_feature랑 매칭?
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features.repeat(b,1,1,1)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens + self.num_hq_token + 1):
            if i < self.num_mask_tokens:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
                
            elif i == self.num_mask_tokens + self.num_hq_token :
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
            
        res_mask_tokens_out = mask_tokens_out[:, i:, :]
        for j in range(self.additional_token):
            hyper_in_list.append(self.output_hypernetworks_additional_mlps[j](res_mask_tokens_out[:, j, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:self.num_mask_tokens] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_sam_hq = (hyper_in[:,self.num_mask_tokens: self.num_mask_tokens + self.num_hq_token, :] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
        masks_sam_aug = (hyper_in[:,self.num_mask_tokens + self.num_hq_token : ] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        
        masks = torch.cat([masks_sam,masks_sam_hq, masks_sam_aug],dim=1)
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
    
    def forward(self, patch_feature):
        batch_num_patches = patch_feature.shape[0]
        patch_feature = self.avgpool(patch_feature).flatten(1) # bs X patchnumber X feature_dim
        #patch_feature = patch_feature \
        patch_feature = self.patch2query(patch_feature) \
            .view(1, batch_num_patches, -1) \
            .repeat_interleave(self.repeat_times, dim=1) \
            .permute(1, 0, 2) \
            .contiguous() 
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
        features = features.flatten(2).permute(2, 0, 1).contiguous()  # hw * bs * dim
        
        proj_feat = self.query_conv(features)
        patches_feat = self.key_conv(patches)
        patches_ca = self.activation(self.dynamic_pattern_conv(patches_feat))
        
        proj_feat = proj_feat.permute(1, 0, 2).contiguous()
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
        features = features.flatten(2).permute(0, 2, 1)  # bs * hw * c
        patches_feat = patches_feat.permute(1, 2, 0)      # bs * c * exemplar_number
        energy = torch.bmm(features, patches_feat)       # bs * hw * exemplar_number

        return energy



class SimilarityWeightedAggregation(nn.Module):
    """
    Implement the multi-head attention with convolution to keep the spatial structure.
    """

    def __init__(self, embed_dim, head, dropout):
        super().__init__()
        self.pool = "max"
        self.pool_size = [1,1]
        self.embed_dim = embed_dim
        self.head = head
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // head
        assert self.head_dim * head == self.embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.in_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.out_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)

    def forward(self, query, keys, values):
        """
        query: 1 x C x H x W
        keys: list of 1 x C x H x W
        values: list of 1 x C x H x W
        """
        h_p, w_p = self.pool_size
        pad = (w_p // 2, w_p // 2, h_p // 2, h_p // 2)
        _, _, h_q, w_q = query.size()

        ##################################################################################
        # calculate similarity (attention)
        ##################################################################################
        query = self.in_conv(query)
        query = query.permute(0, 2, 3, 1).contiguous()
        query = self.norm(query).permute(0, 3, 1, 2).contiguous()
        query = query.contiguous().view(
            self.head, self.head_dim, h_q, w_q
        )  # [head,c,h,w]
        attns_list = []
        key = keys
        if self.pool == "max":
            key = F.adaptive_max_pool2d(key, self.pool_size, return_indices=False)
        else:
            key = F.adaptive_avg_pool2d(key, self.pool_size)
        key = self.in_conv(key)
        key = key.permute(0, 2, 3, 1).contiguous()
        key = self.norm(key).permute(0, 3, 1, 2).contiguous()
        key = key.contiguous().view(
            self.head, self.head_dim, h_p, w_p
        )  # [head,c,h,w]
        attn_list = []
        for q, k in zip(query, key):
            attn = F.conv2d(F.pad(q.unsqueeze(0), pad), k.unsqueeze(0))  # [1,1,h,w]
            attn_list.append(attn)
        attn = torch.cat(attn_list, dim=0)  # [head,1,h,w]
        attns_list.append(attn)
        attns = torch.cat(attns_list, dim=1)  # [head,n,h,w]
        assert list(attns.size()) == [self.head, len(keys), h_q, w_q]

        ##################################################################################
        # score normalization
        ##################################################################################
        attns = attns * float(self.embed_dim * h_p * w_p) ** -0.5  # scaling
        attns = torch.exp(attns)  # [head,n,h,w]
        attns_sn = (
            attns / (attns.max(dim=2, keepdim=True)[0]).max(dim=3, keepdim=True)[0]
        )
        attns_en = attns / attns.sum(dim=1, keepdim=True)
        attns = self.dropout(attns_sn * attns_en)

        ##################################################################################
        # similarity weighted aggregation
        ##################################################################################
        feats = 0
        for idx, value in enumerate(values):
            if self.pool == "max":
                value = F.adaptive_max_pool2d(
                    value, self.pool_size, return_indices=False
                )
            else:
                value = F.adaptive_avg_pool2d(value, self.pool_size)
            attn = attns[:, idx, :, :].unsqueeze(1)  # [head,1,h,w]
            value = self.in_conv(value)
            value = value.contiguous().view(
                self.head, self.head_dim, h_p, w_p
            )  # [head,c,h,w]
            feat_list = []
            for w, v in zip(attn, value):
                feat = F.conv2d(
                    F.pad(w.unsqueeze(0), pad), v.unsqueeze(1).flip(2, 3)
                )  # [1,c,h,w]
                feat_list.append(feat)
            feat = torch.cat(feat_list, dim=0)  # [head,c,h,w]
            feats += feat
        assert list(feats.size()) == [self.head, self.head_dim, h_q, w_q]
        feats = feats.contiguous().view(1, self.embed_dim, h_q, w_q)  # [1,c,h,w]
        feats = self.out_conv(feats)
        return feats