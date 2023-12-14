
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
        # gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        # gray = self.to_tensor(gray)
                
        prompt_w, prompt_h = 32, 32
                
        if self.phase == 'test':
            # prompt_w, prompt_h = random.randint(60, self.image_size // 4 ), random.randint(60, self.image_size // 4 )
            coord_x, coord_y= int(original_width // 2) - (prompt_w//2), int(original_height // 2) - (prompt_h//2)
        else:
            # prompt_w, prompt_h = random.randint(60, self.image_size // 4 ), random.randint(60, self.image_size // 4 )
            coord_x, coord_y = random.randint(0, original_width  - prompt_w*2), random.randint(0, original_height - prompt_h*2)
            
            # coord_x, coord_y = random.randint(0, original_width - self.prompt_w), random.randint(0, original_height - self.prompt_w)
        
        ## define coordinates
        x, y = int(coord_x * ratio_w), int(coord_y * ratio_h)
        
        # normalized_bbox = [int(x * ratio_w), int(y * ratio_h) , int((x + prompt_w) * ratio_w), int((y + prompt_h) * ratio_h)]
        normalized_bbox = [x, y , x + prompt_w, y + prompt_h]
        ## embedding dim
        # normalized_box = np.divide(normalized_bbox, 1024 // 64)
        # x1, y1, x2, y2 = np.around(normalized_box)
        
        # normalized_bbox = [0, 0 , w, w]
        
        masks = []
        mask = Image.open(self.mask[index]).convert('RGB')
        mask = self.image_resize(mask)
        mask = np.array(mask) 
        mask, gt_class = self.mask_normalize(mask, normalized_bbox)
        mask = (mask > 0.5).astype(np.uint8)
        masks.append(mask)
        masks = np.stack(masks, axis=0)

        entire_bboxes = []
        ## mask preprocessing
        # entire_bboxes.append([0, 0 , self.image_size, self.image_size])
        entire_bboxes.append(normalized_bbox)
        entire_bboxes = np.stack(entire_bboxes, axis=0)
            
        prompt_boxes = []
        ## ancor_image
        prompt_boxes.append(normalized_bbox)
        prompt_boxes = np.stack(prompt_boxes, axis=0)
        
        f_name = file_name.split('\\')
        f_name = f_name[2] + '_' + f_name[-1].split('.')[0]
        
        return image, torch.tensor(entire_bboxes), torch.tensor(masks).long(), prompt_boxes, f_name, gt_class
    
    
    @classmethod
    def collate_fn(cls, batch):
        images, bboxes, masks, prompt_boxs, f_name, gt_class = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, bboxes, masks, prompt_boxs, f_name, gt_class
    
    
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