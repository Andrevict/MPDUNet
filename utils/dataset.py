from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from torchvision import transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, image_list, mask_list, transform=None):
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # print(f'正在读取: {self.image_list[idx]}')
        image = Image.open(self.image_list[idx]).convert("RGB")
        mask = Image.open(self.mask_list[idx]).convert("L")  # L 表示灰度图

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # 将掩码转换为二值图（根据你的任务调整）
        mask = (mask > 0.5).float()

        return image, mask
