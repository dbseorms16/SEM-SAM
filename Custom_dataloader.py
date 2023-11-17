
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import cv2

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from glob import glob
import random
# coco mask style dataloader
class Custom_Dataset(Dataset):
    def __init__(self, data_root, phase, image_size):
        self.data_root = data_root
        self.phase = phase
        self.image_size = image_size
        self.x, self.mask = self.load_dataset_folder()
        
        # TODO: use ResizeLongestSide and pad to square
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_resize = transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR)
        
        self.prompt_w = 30

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        ## image preprocessing
        x = self.x[index]
        image = Image.open(x).convert("RGB")
        original_width, original_height = image.width, image.height
        ratio_h = self.image_size / image.height
        ratio_w = self.image_size / image.width
        image = self.image_resize(image)
        image = self.to_tensor(image)
        image = self.normalize(image)

        
        # 30 pixels ranmdom box prompt generation
        
        box_x, box_y = random.randint(0, original_width - self.prompt_w), random.randint(0, original_height - self.prompt_w)
        x, y, w = box_x, box_y, self.prompt_w 
        bbox = [x * ratio_w, y * ratio_h, (x + w) * ratio_w, (y + w) * ratio_h]

        
        bboxes = []
        masks = []

        ## mask preprocessing
        mask = Image.open(self.mask[index]).convert('RGB')
        mask = self.mask_normalize(mask, bbox)

        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.uint8)
            
        bboxes.append(bbox)
        masks.append(mask)

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        return image, torch.tensor(bboxes), torch.tensor(masks).long()
    
    @classmethod
    def collate_fn(cls, batch):
        images, bboxes, masks = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, bboxes, masks
    
    
    def load_dataset_folder(self):
        #TODO folder name to glob

        class_names = ['300W', '350W', '400W', '450W', '500W']
        
        x, mask = [], []
        
        for class_name in class_names:
            img_dir = os.path.join(self.data_root, self.phase, class_name, 'img')
            mask_dir = os.path.join(self.data_root, self.phase, class_name, 'mask')

            img_fpath_list = sorted([os.path.join(img_dir, f)
                                    for f in os.listdir(img_dir)
                                    if f.endswith('.bmp') or f.endswith('.jpg')])
            x.extend(img_fpath_list)

            mask_fpath_list = sorted([os.path.join(mask_dir, f)
                                    for f in os.listdir(mask_dir)
                                    if f.endswith('.png')])
            mask.extend(mask_fpath_list)


        assert len(x) == len(mask), 'number of x and mask should be same'

        return list(x), list(mask)
    
    def mask_normalize(self, mask, bbox):
        print(mask)
        normalized_mask = mask
        
        return normalized_mask