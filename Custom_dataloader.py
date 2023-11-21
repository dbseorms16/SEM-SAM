
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
        
        self.prompt_w = 128

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        ## image preprocessing
        file_name = self.x[index]
        img = Image.open(file_name).convert("RGB")
        original_width, original_height = img.width, img.height
        ratio_h = self.image_size / img.height
        ratio_w = self.image_size / img.width
        image = self.image_resize(img)
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        
        if self.phase == 'test':
            coord_x, coord_y = int(original_width // 2) - (self.prompt_w//2), int(original_height // 2) - (self.prompt_w//2)
        else:
            coord_x, coord_y = random.randint(0, original_width * ratio_w - self.prompt_w - 1 ), random.randint(0, original_height * ratio_h - self.prompt_w -1 )
            # coord_x, coord_y = random.randint(0, original_width - self.prompt_w), random.randint(0, original_height - self.prompt_w)
        
        ## define coordinates
        x, y, w = coord_x, coord_y, self.prompt_w 
        coord = [x, y, x + w, y + w]
        
        masks = []
        mask = Image.open(self.mask[index]).convert('RGB')
        mask = self.image_resize(mask)
        mask = np.array(mask) 
        mask, gt_class = self.mask_normalize(mask, coord)
        mask = (mask > 0.5).astype(np.uint8)
        masks.append(mask)
        masks = np.stack(masks, axis=0)

        bboxes = []
        ## mask preprocessing
        normalized_bbox = [x , y , x  + w, y + w]
        bboxes.append(normalized_bbox)
        bboxes = np.stack(bboxes, axis=0)
            
        prompt_imgs = []
        ## ancor_image
        prompt_img = image[:, int(y): int(y + w), int(x): int(x + w)]
        
        prompt_imgs.append(prompt_img)
        prompt_imgs = torch.stack(prompt_imgs, axis=0)
        
        f_name = file_name.split('\\')
        f_name = f_name[2] + '_' + f_name[-1].split('.')[0]
        
        
        return image, torch.tensor(bboxes), torch.tensor(masks).long(), prompt_imgs, f_name, gt_class
    
    
    @classmethod
    def collate_fn(cls, batch):
        images, bboxes, masks, prompt_imgs, f_name, gt_class = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, bboxes, masks, prompt_imgs, f_name, gt_class
    
    
    def load_dataset_folder(self):
        #TODO folder name to glob

        # class_names = ['300W', '350W', '400W', '450W', '500W']
        class_names = ['300W', '350W', '400W', '450W', '500W', 'X500W']
        
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
        
        ## add purple 
        m_R, m_B = mask[:,:,0], mask[:,:,2]
        m_P = ((m_R == 100) & (m_B == 100)) * 255
        mask = np.insert(mask, 3, m_P, axis=2)
        
        
        prompt = mask[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2]), :]
        R, G, B, P = prompt[:,:,0], prompt[:,:,1], prompt[:,:,2], prompt[:,:,3]
        
        r_count = self.count_pixel_label(R)
        g_count = self.count_pixel_label(G)
        b_count = self.count_pixel_label(B)
        p_count = self.count_pixel_label(P)
        
        list = [r_count, g_count, b_count, p_count]
        nums = sorted(list, reverse=True)
        index = list.index(nums[0])     
        normalized_mask = (mask[:, :, index] == 255) * 255. 

        return normalized_mask, index
    
    def count_pixel_label(self, label):
        
        pixel_values, count = np.unique(label, return_counts=True)
        
        if 255 in pixel_values :
            result = count[-1]
        else:
            result = 0
                
        return result