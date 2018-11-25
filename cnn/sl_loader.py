from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
import glob

class SL_Dataset_Train(Dataset):
    def __init__(self, image_dir, label_file, image_transforms=None):
        
        assert(os.path.exists(label_file))
        assert(os.path.exists(image_dir))
        
        self.data = self.gather_files(image_dir, label_file)
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        img = Image.open(self.data[index][0])
        for t in self.image_transforms: img = t(img)
        img = torch.from_numpy(np.asarray(img)).float()

        label = self.data[index][1]

        return [img, label]

    def gather_files(self, image_dir, label_file):
        #for each label_file line, take the image, assign the labels
        #seperate label probs by space only
        
        data = []

        with open(label_file, 'r') as f:
            for i,line in enumerate(f):
                line_data = line.replace('\n','').split(' ')
                assert(len(line_data) == 25)
                
                img_file = line_data.pop(0)
                img_file = os.path.join(image_dir, img_file)
                
                line_data = [float(j) for j in line_data]
                line_data = torch.Tensor(line_data).long()

                assert(os.path.exists(img_file))

                data.append((img_file,line_data))

        assert(data != [])
        return data
    
class SL_Dataset_Test(Dataset):
    def __init__(self, image_dir, image_ext, image_transforms=None):
        
        assert(os.path.exists(image_dir))
        
        self.data = self.gather_files(image_dir, image_ext)
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        img = Image.open(self.data[index])
        for t in self.image_transforms: img = t(img)
        img = torch.from_numpy(np.asarray(img)).float()

        return img

    def gather_files(self, image_dir, image_ext):
        
        assert(os.path.exists(image_dir))
        data = glob.glob(os.path.join(image_dir,('*' + image_ext)))

        return data