from torch.utils.data import Dataset, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


class FruitDatabase(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


def get_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])


def build_datasets(data_dir, train_ratio=0.7, val_ratio=0.15):
    dataset = FruitDatabase(data_dir, transform=get_transforms())

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    return dataset, train_ds, val_ds, test_ds