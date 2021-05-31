import os
import os.path as osp
from PIL import Image
import torch
from torch.utils import data
from torchvision import datasets, transforms



class MyDataLoader(data.Dataset):
    """
        when define the son class of torch.utils.data.Dataset, len and getitem function must reload.
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
    """
    def __init__(self, img_root, txt_file, transforms=None, isCaseW=False, train=True):

        self.img_list = []
        self.labels = []
        self.img_root = img_root
        self.isCaseW = isCaseW
        self.read_txt_file(txt_file)
        self.transforms = transforms
        
        
    def __getitem__(self, index):
        """
            return one image and label
        """
        img_path = osp.join(self.img_root, self.img_list[index]) 
        img = Image.open(img_path)
        img = self.transforms(img)
        label = self.labels[index]
        return img, label
    
    
    def __len__(self, ):
        return len(self.img_list)

    def read_txt_file(self, txt_file):
        with open(txt_file, "r") as fr:
            for line in fr:
                img_name, cls_name = line.split()
                temp_label = int(cls_name)
                self.img_list.append(img_name)
                self.labels.append(temp_label)