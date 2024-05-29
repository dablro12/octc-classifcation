from torch.utils.data import Dataset, BatchSampler
import os 
from PIL import Image
import numpy as np
import random

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform = None): #transform 가지고올거있으면 가지고 오기 
        self.root_dir = root_dir 
        self.transform = transform
        self.labels = []
        self.image_paths = []
        
        # data 읽어서 labels와 image_path에 저장
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for filename in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, filename)
                    self.labels.append(float(label))
                    self.image_paths.append(file_path)
                
        
    def __len__(self):
        return len(self.labels) 
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        #이미지 open to PIL : pytorch는 PIL 선호
        # opencv bgr -> rgb 로 변환
        image = Image.open(image_path).convert('RGB')
        
        # 위아래 제외하고 crop 해놓기 -> 원본과 같은사이즈로
        # crop_img = image.crop((0, 120, image.width, image.height - 50))
        # image = Image.new("RGB", image.size, (0,0,0))
        # image.paste(crop_img, (0,150)) 
        
        # transform 
        if self.transform:
            image = self.transform(image)
        
        return image, label, image_path
    
    def getlabels(self):
        return self.labels
    
    
class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # class 에 따라 구분
        self.class0_indices = [i for i, (_, label) in enumerate(dataset) if label == 0]
        self.class1_indices = [i for i, (_, label) in enumerate(dataset) if label == 1]

    def __iter__(self):
        random.shuffle(self.class0_indices)
        random.shuffle(self.class1_indices)

        # 반반씩 뽑기 위한 배치 사이즈 조정
        half_batch = self.batch_size // 2

        for i in range(0, min(len(self.class0_indices), len(self.class1_indices)), half_batch):
            batch_indices = []
            batch_indices.extend(self.class0_indices[i:i + half_batch])
            batch_indices.extend(self.class1_indices[i:i + half_batch])
            random.shuffle(batch_indices)  # 배치 내부 셔플
            yield batch_indices

    def __len__(self):
        return min(len(self.class0_indices), len(self.class1_indices)) // (self.batch_size // 2)
