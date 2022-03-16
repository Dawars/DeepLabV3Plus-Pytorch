import os

from PIL import Image
from torch.utils import data


class ListDataset(data.Dataset):

    def __init__(self, root, file_list_path, transform=None):
        self.root = os.path.expanduser(root)
        self.file_list_path = file_list_path
        self.transform = transform

        self.images = []
        with open(file_list_path) as f:
            for line in f.readlines():
                filename = line.strip()
                path = os.path.join(root, filename)
                self.images.append(path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        out = Image.open(self.images[index]).convert('RGB')
        if self.transform:
            out = self.transform(out)
        return out
