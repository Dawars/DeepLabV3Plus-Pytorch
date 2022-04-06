import os

from PIL import Image
from torch.utils import data


class ListDataset(data.Dataset):

    def __init__(self, image_root, file_list_path, transform=None):
        self.image_root = os.path.expanduser(image_root)
        self.file_list_path = file_list_path
        self.transform = transform

        self.images = []
        with open(file_list_path) as f:
            for line in f.readlines():
                filename = line.strip()
                path = os.path.join(image_root, filename)
                self.images.append(path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        out = Image.open(f"{self.images[index]}.png").convert('RGB')
        filename = os.path.relpath(self.images[index], self.image_root)
        if self.transform:
            out = self.transform(out)
        return out, filename


if __name__ == '__main__':
    dataset = ListDataset('/mnt/hdd/datasets/facade/ZuBuD/ZuBuD/png-ZuBuD',)