from logging import root
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
import cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTurbVideo(Dataset):
    def __init__(self, root_dir, num_frames=12, patch_size=None, noise=None, is_train=True):
        super(DataLoaderTurbVideo, self).__init__()
        self.num_frames = num_frames
        self.turb_list = []
        self.blur_list = []
        self.gt_list = []
        for v in os.listdir(os.path.join(root_dir, 'gt')):
            self.gt_list.append(os.path.join(root_dir, 'gt', v))
            self.turb_list.append(os.path.join(root_dir, 'turb', v))
            self.blur_list.append(os.path.join(root_dir, 'blur', v))

        self.ps = patch_size
        self.sizex = len(self.gt_list)  # get the size of target
        self.train = is_train
        self.noise = noise

    def __len__(self):
        return self.sizex

    def _inject_noise(self, img, noise):
        noise = (noise**0.5)*torch.randn(img.shape)
        out = img + noise
        return out.clamp(0,1)
        
    def _fetch_chunk_val(self, idx):
        ps = self.ps
        turb_vid = cv2.VideoCapture(self.turb_list[idx])
        blur_vid = cv2.VideoCapture(self.blur_list[idx])
        gt_vid = cv2.VideoCapture(self.gt_list[idx])
        h, w = int(gt_vid.get(4)), int(gt_vid.get(3))
        total_frames = int(gt_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            print('no enough frame in video ' + self.gt_list[idx])
        start_frame_id = (total_frames-self.num_frames) // 2
        
        # load frames from video
        gt_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        turb_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        blur_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        tar_imgs = [gt_vid.read()[1] for i in range(self.num_frames)]
        turb_imgs = [turb_vid.read()[1] for i in range(self.num_frames)]
        blur_imgs = [blur_vid.read()[1] for i in range(self.num_frames)]
        turb_vid.release()
        blur_vid.release()
        gt_vid.release()
        
        tar_imgs =  [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in tar_imgs]
        turb_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in turb_imgs]
        blur_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in blur_imgs]     
        
        if ps > 0:
            padw = ps-w if w<ps else 0
            padh = ps-h if h<ps else 0
            if padw!=0 or padh!=0:
                blur_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in blur_imgs]
                turb_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in turb_imgs]   
                tar_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in tar_imgs]   

            blur_imgs = [TF.to_tensor(img) for img in blur_imgs]
            turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
            tar_imgs  = [TF.to_tensor(img) for img in tar_imgs]

            hh, ww = tar_imgs[0].shape[1], tar_imgs[0].shape[2]
            
            rr     = (hh-ps) // 2
            cc     = (ww-ps) // 2
            # Crop patch
            blur_imgs = [img[:, rr:rr+ps, cc:cc+ps] for img in blur_imgs]
            turb_imgs = [img[:, rr:rr+ps, cc:cc+ps] for img in turb_imgs]
            tar_imgs  = [img[:, rr:rr+ps, cc:cc+ps] for img in tar_imgs]
        else:
            blur_imgs = [TF.to_tensor(img) for img in blur_imgs]
            turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
            tar_imgs  = [TF.to_tensor(img) for img in tar_imgs]
        
        if self.noise:
            noise_level = self.noise * random.random()
            blur_imgs = [self._inject_noise(img, noise_level) for img in blur_imgs]
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]
        return blur_imgs, turb_imgs, tar_imgs
                         
    def _fetch_chunk_train(self, idx):
        ps = self.ps
        turb_vid = cv2.VideoCapture(self.turb_list[idx])
        blur_vid = cv2.VideoCapture(self.blur_list[idx])
        gt_vid = cv2.VideoCapture(self.gt_list[idx])
        h, w = int(gt_vid.get(4)), int(gt_vid.get(3))
        total_frames = int(gt_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            print('no enough frame in video ' + self.gt_list[idx])
            # return self._fetch_chunk_train(idx + 1)
        start_frame_id = random.randint(0, total_frames-self.num_frames)
        
        # load frames from video
        gt_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        turb_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        blur_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        tar_imgs = [gt_vid.read()[1] for i in range(self.num_frames)]
        turb_imgs = [turb_vid.read()[1] for i in range(self.num_frames)]
        blur_imgs = [blur_vid.read()[1] for i in range(self.num_frames)]
        if tar_imgs[0] is None:
            print(self.gt_list[idx])
        # tar_imgs =  [Image.fromarray(img).convert('RGB') for img in tar_imgs]
        # turb_imgs = [Image.fromarray(img).convert('RGB') for img in turb_imgs]
        # blur_imgs = [Image.fromarray(img).convert('RGB') for img in blur_imgs]
        tar_imgs =  [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in tar_imgs]
        turb_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in turb_imgs]
        blur_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in blur_imgs]        
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0
        if padw!=0 or padh!=0:
            blur_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in blur_imgs]
            turb_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in turb_imgs]   
            tar_imgs  = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in tar_imgs]   
        
        aug    = random.randint(0, 2)
        if aug == 1:
            blur_imgs = [TF.adjust_gamma(img, 1) for img in blur_imgs]
            turb_imgs = [TF.adjust_gamma(img, 1) for img in turb_imgs]
            tar_imgs  = [TF.adjust_gamma(img, 1) for img in tar_imgs]   
            
        aug    = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            blur_imgs = [TF.adjust_saturation(img, sat_factor) for img in blur_imgs]
            turb_imgs = [TF.adjust_saturation(img, sat_factor) for img in turb_imgs]
            tar_imgs  = [TF.adjust_saturation(img, sat_factor) for img in tar_imgs]
            
        hh, ww = h, w

        enlarge_factor = random.choice([0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.2, 1,2, 1.5, 1.8, 2])
        crop_size = ps * enlarge_factor
        crop_size = min(hh, ww, crop_size)
        hcro = int(crop_size * random.uniform(1.1, 0.9))
        wcro = int(crop_size * random.uniform(1.1, 0.9))
        hcro = min(hcro, hh)
        wcro = min(wcro, ww)
        rr   = random.randint(0, hh-hcro)
        cc   = random.randint(0, ww-wcro)
        
        # Crop patch
        blur_imgs = [TF.resize(img.crop((cc, rr, cc+wcro, rr+hcro)), (ps, ps)) for img in blur_imgs]
        turb_imgs = [TF.resize(img.crop((cc, rr, cc+wcro, rr+hcro)), (ps, ps)) for img in turb_imgs]
        tar_imgs  = [TF.resize(img.crop((cc, rr, cc+wcro, rr+hcro)), (ps, ps)) for img in tar_imgs]
        
        blur_imgs = [TF.to_tensor(img) for img in blur_imgs]
        turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
        tar_imgs  = [TF.to_tensor(img) for img in tar_imgs]

        if self.noise:
            noise_level = self.noise * random.random()
            blur_imgs = [self._inject_noise(img, noise_level) for img in blur_imgs]
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]
            
        aug    = random.randint(0, 8)
        # Data Augmentations
        if aug==1:
            blur_imgs = [img.flip(1) for img in blur_imgs]
            turb_imgs = [img.flip(1) for img in turb_imgs]
            tar_imgs  = [img.flip(1) for img in tar_imgs]
        elif aug==2:
            blur_imgs = [img.flip(2) for img in blur_imgs]
            turb_imgs = [img.flip(2) for img in turb_imgs]
            tar_imgs  = [img.flip(2) for img in tar_imgs]
        elif aug==3:
            blur_imgs = [torch.rot90(img, dims=(1,2)) for img in blur_imgs]
            turb_imgs = [torch.rot90(img, dims=(1,2)) for img in turb_imgs]
            tar_imgs  = [torch.rot90(img, dims=(1,2)) for img in tar_imgs]
        elif aug==4:
            blur_imgs = [torch.rot90(img,dims=(1,2), k=2) for img in blur_imgs]
            turb_imgs = [torch.rot90(img,dims=(1,2), k=2) for img in turb_imgs]
            tar_imgs  = [torch.rot90(img,dims=(1,2), k=2) for img in tar_imgs]
        elif aug==5:
            blur_imgs = [torch.rot90(img,dims=(1,2), k=3) for img in blur_imgs]
            turb_imgs = [torch.rot90(img,dims=(1,2), k=3) for img in turb_imgs]
            tar_imgs  = [torch.rot90(img,dims=(1,2), k=3) for img in tar_imgs]
        elif aug==6:
            blur_imgs = [torch.rot90(img.flip(1), dims=(1,2)) for img in blur_imgs]
            turb_imgs = [torch.rot90(img.flip(1), dims=(1,2)) for img in turb_imgs]
            tar_imgs  = [torch.rot90(img.flip(1), dims=(1,2)) for img in tar_imgs]
        elif aug==7:
            blur_imgs = [torch.rot90(img.flip(2), dims=(1,2)) for img in blur_imgs]
            turb_imgs = [torch.rot90(img.flip(2), dims=(1,2)) for img in turb_imgs]
            tar_imgs  = [torch.rot90(img.flip(2), dims=(1,2)) for img in tar_imgs]
        return blur_imgs, turb_imgs, tar_imgs
                           
    def __getitem__(self, index):
        index_ = index % self.sizex
        if self.train:
            blur_imgs, turb_imgs, tar_imgs = self._fetch_chunk_train(index_)
        else:
            blur_imgs, turb_imgs, tar_imgs = self._fetch_chunk_val(index_)
        return torch.stack(blur_imgs, dim=0), torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0)
        
      
class DataLoaderTurbImage(Dataset):
    # equal to previous dynamic
    def __init__(self, rgb_dir, num_frames=12, total_frames=60, im_size=None, noise=None, other_turb=None, is_train=True, img_filter = ""):
        super(DataLoaderTurbImage, self).__init__()
        self.num_frames = num_frames
        self.img_list = [os.path.join(rgb_dir, d) for d in os.listdir(rgb_dir) if (len(img_filter)<0 or img_filter in d )]
        
                
        self.img_list = [d for d in self.img_list if len(os.listdir(d))>=3]
        print("Generating img filter for ", img_filter, len( self.img_list))
        self.total_frames = total_frames
        self.ps = im_size
        self.sizex = len(self.img_list)  # get the size of target
        self.train = is_train
        self.noise = noise
        self.other_turb = other_turb

    def __len__(self):
        return self.sizex

    def _inject_noise(self, img, noise_level):
        noise = (noise_level**0.5)*torch.randn(img.shape)
        out = img + noise
        return out.clamp(0,1)
        
    def _fetch_chunk_val(self, seq_path, frame_idx):
        ps = self.ps
        blur_img_list = [os.path.join(seq_path, 'blur', '{:d}.jpg'.format(n)) for n in frame_idx]
        if self.other_turb:
            turb_img_list = [os.path.join(self.other_turb, 'turb_out', '{:d}.jpg'.format(n)) for n in frame_idx]
        else:
            turb_img_list = [os.path.join(seq_path, 'turb', '{:d}.jpg'.format(n)) for n in frame_idx]
        blur_imgs = [Image.open(p) for p in blur_img_list]
        turb_imgs = [Image.open(p) for p in turb_img_list]
        tar_img = Image.open(os.path.join(seq_path, 'gt.jpg'))
        w,h = tar_img.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0
        if padw!=0 or padh!=0:
            blur_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in blur_imgs]
            turb_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in turb_imgs]   
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')
                      
        blur_imgs = [TF.to_tensor(img) for img in blur_imgs]
        turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]
        
        rr     = (hh-ps) // 2
        cc     = (ww-ps) // 2
         # Crop patch
        blur_imgs = [img[:, rr:rr+ps, cc:cc+ps] for img in blur_imgs]
        turb_imgs = [img[:, rr:rr+ps, cc:cc+ps] for img in turb_imgs]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]
        if self.noise:
            noise_level = self.noise * random.random()
            blur_imgs = [self._inject_noise(img, noise_level) for img in blur_imgs]
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]
        return blur_imgs, turb_imgs, [tar_img]*self.num_frames
                         
    def _fetch_chunk_train(self, seq_path, frame_idx):
        ps = self.ps
        blur_img_list = [os.path.join(seq_path, 'blur', '{:d}.jpg'.format(n)) for n in frame_idx]
        if self.other_turb:
            turb_img_list = [os.path.join(self.other_turb, 'turb_out', '{:d}.jpg'.format(n)) for n in frame_idx]
        else:
            turb_img_list = [os.path.join(seq_path, 'turb', '{:d}.jpg'.format(n)) for n in frame_idx]
        blur_imgs = [Image.open(p) for p in blur_img_list]
        turb_imgs = [Image.open(p) for p in turb_img_list]
        tar_img = Image.open(os.path.join(seq_path, 'gt.jpg'))
        w,h = tar_img.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0
        if padw!=0 or padh!=0:
            blur_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in blur_imgs]
            turb_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in turb_imgs]   
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')
        
        aug    = 0#random.randint(0, 2)
        if aug == 1:
            blur_imgs = [TF.adjust_gamma(img, 1) for img in blur_imgs]
            turb_imgs = [TF.adjust_gamma(img, 1) for img in turb_imgs]
            tar_img = TF.adjust_gamma(tar_img, 1)
            
        aug    = 0#random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            blur_imgs = [TF.adjust_saturation(img, sat_factor) for img in blur_imgs]
            turb_imgs = [TF.adjust_saturation(img, sat_factor) for img in turb_imgs]
            tar_img = TF.adjust_saturation(tar_img, sat_factor)
            
        blur_imgs = [TF.to_tensor(img) for img in blur_imgs]
        turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        enlarge_factor = 1#random.choice([0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.2, 1,2, 1.5, 1.8, 2])
        crop_size = ps * enlarge_factor
        crop_size = min(hh, ww, crop_size)
        # hcro = int(crop_size * random.uniform(1.1, 0.9))
        # wcro = int(crop_size * random.uniform(1.1, 0.9))
        hcro = hh
        wcro = ww
        rr   = 0#random.randint(0, hh-hcro)
        cc   = 0#random.randint(0, ww-wcro)
        
        # Crop patch
        blur_imgs = [TF.resize(img[:, rr:rr+hcro, cc:cc+wcro], (ps, ps)) for img in blur_imgs]
        turb_imgs = [TF.resize(img[:, rr:rr+hcro, cc:cc+wcro], (ps, ps)) for img in turb_imgs]
        tar_img = TF.resize(tar_img[:, rr:rr+hcro, cc:cc+wcro], (ps, ps))

        if False and self.noise:
            noise_level = self.noise * random.random()
            blur_imgs = [self._inject_noise(img, noise_level) for img in blur_imgs]
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]
            
        aug    = 0 #random.randint(0, 8)
        # Data Augmentations
        if aug==1:
            blur_imgs = [img.flip(1) for img in blur_imgs]
            turb_imgs = [img.flip(1) for img in turb_imgs]
            tar_img = tar_img.flip(1)
        elif aug==2:
            blur_imgs = [img.flip(2) for img in blur_imgs]
            turb_imgs = [img.flip(2) for img in turb_imgs]
            tar_img = tar_img.flip(2)
        elif aug==3:
            blur_imgs = [torch.rot90(img, dims=(1,2)) for img in blur_imgs]
            turb_imgs = [torch.rot90(img, dims=(1,2)) for img in turb_imgs]
            tar_img = torch.rot90(tar_img, dims=(1,2))
        elif aug==4:
            blur_imgs = [torch.rot90(img,dims=(1,2), k=2) for img in blur_imgs]
            turb_imgs = [torch.rot90(img,dims=(1,2), k=2) for img in turb_imgs]
            tar_img = torch.rot90(tar_img, dims=(1,2), k=2)
        elif aug==5:
            blur_imgs = [torch.rot90(img,dims=(1,2), k=3) for img in blur_imgs]
            turb_imgs = [torch.rot90(img,dims=(1,2), k=3) for img in turb_imgs]
            tar_img = torch.rot90(tar_img, dims=(1,2), k=3)
        elif aug==6:
            blur_imgs = [torch.rot90(img.flip(1), dims=(1,2)) for img in blur_imgs]
            turb_imgs = [torch.rot90(img.flip(1), dims=(1,2)) for img in turb_imgs]
            tar_img = torch.rot90(tar_img.flip(1), dims=(1,2))
        elif aug==7:
            blur_imgs = [torch.rot90(img.flip(2), dims=(1,2)) for img in blur_imgs]
            turb_imgs = [torch.rot90(img.flip(2), dims=(1,2)) for img in turb_imgs]
            tar_img = torch.rot90(tar_img.flip(2), dims=(1,2))
        return blur_imgs, turb_imgs, [tar_img]*self.num_frames
                           
    def __getitem__(self, index):
        index_ = index % self.sizex
        start_frame_id = random.randint(0, self.total_frames-self.num_frames)
        frame_idx = [i for i in range(start_frame_id, start_frame_id+self.num_frames)]
        seq_path = self.img_list[index_]
        if self.train:
            blur_imgs, turb_imgs, tar_imgs = self._fetch_chunk_train(seq_path, frame_idx)
        else:
            blur_imgs, turb_imgs, tar_imgs = self._fetch_chunk_val(seq_path, frame_idx)
        return torch.stack(blur_imgs, dim=0), torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0)
        
      
class DataLoaderTurbImageTest(Dataset):
    # equal to previous dynamic
    def __init__(self, rgb_dir, num_frames=48, total_frames=50, im_size=None, noise=None, is_train=True, start_frame=0):
        super(DataLoaderTurbImageTest, self).__init__()
        self.num_frames = num_frames
        self.img_list = [os.path.join(rgb_dir, d) for d in os.listdir(rgb_dir)]
        self.img_list = [d for d in self.img_list if len(os.listdir(d))>=3]
        self.total_frames = total_frames
        self.ps = im_size
        self.sizex = len(self.img_list)  # get the size of target
        self.train = is_train
        self.noise = noise
        self.start_frame = start_frame

    def __len__(self):
        return self.sizex
                         
    def _fetch_chunk(self, seq_path, frame_idx):
        turb_img_list = [os.path.join(seq_path, 'turb', '{:d}.jpg'.format(n)) for n in frame_idx]
        turb_imgs = [Image.open(p) for p in turb_img_list]
        tar_img = Image.open(os.path.join(seq_path, 'gt.jpg'))
        turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
        tar_img = TF.to_tensor(tar_img)
        return turb_imgs, tar_img
                           
    def __getitem__(self, index):
        index_ = index % self.sizex
        frame_idx = [i for i in range(self.start_frame, self.start_frame + self.num_frames)]
        seq_path = self.img_list[index_]
        turb_imgs, tar_imgs = self._fetch_chunk(seq_path, frame_idx)
        return torch.stack(turb_imgs, dim=0), tar_imgs, seq_path

class DataLoaderBlurImageTest(Dataset):
    # equal to previous dynamic
    def __init__(self, rgb_dir, blur_dir, num_frames=48, total_frames=48, im_size=None, noise=None, is_train=True):
        super(DataLoaderBlurImageTest, self).__init__()
        self.num_frames = num_frames
        self.img_list = [os.path.join(rgb_dir, d) for d in os.listdir(rgb_dir)]
        self.img_list = [d for d in self.img_list if len(os.listdir(d))>=3]
        self.blur_dir = blur_dir
        self.total_frames = total_frames
        self.ps = im_size
        self.sizex = len(self.img_list)  # get the size of target
        self.train = is_train
        self.noise = noise

    def __len__(self):
        return self.sizex
                         
    def _fetch_chunk(self, seq_path, frame_idx):
        img_name = os.path.split(seq_path)[-1]
        blur_img_list = [os.path.join(self.blur_dir, img_name, 'turb_out', '{:d}.jpg'.format(n)) for n in frame_idx]
        blur_imgs = [Image.open(p) for p in blur_img_list]
        tar_img = Image.open(os.path.join(seq_path, 'gt.jpg'))
        blur_imgs = [TF.to_tensor(img) for img in blur_imgs]
        tar_img = TF.to_tensor(tar_img)
        return blur_imgs, tar_img
                           
    def __getitem__(self, index):
        index_ = index % self.sizex
        frame_idx = [i for i in range(self.num_frames)]
        seq_path = self.img_list[index_]
        turb_imgs, tar_imgs = self._fetch_chunk(seq_path, frame_idx)
        return torch.stack(turb_imgs, dim=0), tar_imgs, seq_path


class DataLoaderTurbVideoTest(Dataset):
    def __init__(self, root_dir, result_dir, max_frame=120, patch_unit=16):
        super(DataLoaderTurbVideoTest, self).__init__()
        self.turb_list = []
        self.blur_list = []
        self.gt_list = []
        for v in os.listdir(os.path.join(root_dir, 'gt')):
            if not v in os.listdir(result_dir):
                self.gt_list.append(os.path.join(root_dir, 'gt', v))
                self.turb_list.append(os.path.join(root_dir, 'turb', v))

        self.pu = patch_unit
        self.sizex = len(self.gt_list)  # get the size of target
        self.max_frame = 120


    def __len__(self):
        return self.sizex
        
    def _fetch_chunk(self, idx):
        pu = self.pu
        turb_vid = cv2.VideoCapture(self.turb_list[idx])
        gt_vid = cv2.VideoCapture(self.gt_list[idx])
        h, w = int(gt_vid.get(4)), int(gt_vid.get(3))
        fps = int(turb_vid.get(5))
        total_frames = int(gt_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # load frames from video
        load_frames = min(total_frames, self.max_frame)
        tar_imgs = [gt_vid.read()[1] for i in range(load_frames)]
        turb_imgs = [turb_vid.read()[1] for i in range(load_frames)]
        turb_vid.release()
        gt_vid.release()
        
        tar_imgs =  [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in tar_imgs]
        turb_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in turb_imgs]
        
        padh, padw = 0, 0
        if h % pu != 0:
            padh = pu - h % pu
        if w % pu != 0:
            padw = pu - w % pu
        if padh + padw > 0:
            # left, top, right and bottom
            turb_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in turb_imgs]   
            tar_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in tar_imgs]   

        turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
        tar_imgs  = [TF.to_tensor(img) for img in tar_imgs]
        return turb_imgs, tar_imgs, h, w, load_frames, fps
                         
 
    def __getitem__(self, index):
        index_ = index % self.sizex
        turb_imgs, tar_imgs, h, w, load_frames, fps = self._fetch_chunk(index_)
        return torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0), h, w, load_frames, fps, self.turb_list[index_]

class DataLoaderBlurVideoTest(Dataset):
    def __init__(self, root_dir, result_dir, max_frame=120, patch_unit=16):
        super(DataLoaderBlurVideoTest, self).__init__()
        self.turb_list = []
        self.blur_list = []
        self.gt_list = []
        for v in os.listdir(os.path.join(root_dir, 'gt')):
            if not v in os.listdir(result_dir):
                self.gt_list.append(os.path.join(root_dir, 'gt', v))
                self.turb_list.append(os.path.join(root_dir, 'turb', v))

        self.pu = patch_unit
        self.sizex = len(self.gt_list)  # get the size of target
        self.max_frame = 120


    def __len__(self):
        return self.sizex
        
    def _fetch_chunk(self, idx):
        pu = self.pu
        turb_vid = cv2.VideoCapture(self.turb_list[idx])
        gt_vid = cv2.VideoCapture(self.gt_list[idx])
        h, w = int(gt_vid.get(4)), int(gt_vid.get(3))
        fps = int(turb_vid.get(5))
        total_frames = int(gt_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # load frames from video
        load_frames = min(total_frames, self.max_frame)
        tar_imgs = [gt_vid.read()[1] for i in range(load_frames)]
        turb_imgs = [turb_vid.read()[1] for i in range(load_frames)]
        turb_vid.release()
        gt_vid.release()
        
        tar_imgs =  [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in tar_imgs]
        turb_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in turb_imgs]
        
        padh, padw = 0, 0
        if h % pu != 0:
            padh = pu - h % pu
        if w % pu != 0:
            padw = pu - w % pu
        if padh + padw > 0:
            # left, top, right and bottom
            turb_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in turb_imgs]   
            tar_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in tar_imgs]   

        turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
        tar_imgs  = [TF.to_tensor(img) for img in tar_imgs]
        return turb_imgs, tar_imgs, h, w, load_frames, fps
                         
 
    def __getitem__(self, index):
        index_ = index % self.sizex
        turb_imgs, tar_imgs, h, w, load_frames, fps = self._fetch_chunk(index_)
        return torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0), h, w, load_frames, fps, self.turb_list[index_]
    

def get_tmt_loader(dataset_root, n_frames = 1, mode = 'train',  patch_size = 16, batch_size = 8, num_workers = 2, img_filter = "", **kwargs):
    assert mode in ['train', 'val'], "mode must be 'train' or 'val'"

    if mode == 'train':
        dataset = DataLoaderTurbImage(
            rgb_dir=dataset_root,
            num_frames=n_frames,
            total_frames=50,
            im_size=patch_size,
            noise=0,
            is_train=True,
            img_filter = img_filter
        )
    else:
        dataset = DataLoaderTurbImage(
            rgb_dir=dataset_root,
            num_frames=n_frames,
            total_frames=50,
            im_size=patch_size,
            noise=0,
            is_train=False
        )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )

    return loader
