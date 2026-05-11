"""Binary MNIST dataset utilities."""

from torch.utils.data import Dataset
from torchvision import datasets


class BinaryMNISTDataset(Dataset):
    """MNIST wrapper that maps digits to even/odd classes.

    Class mapping:
        0 -> even
        1 -> odd
    """

    classes = ("even", "odd")

    def __init__(self, root, train=True, download=True, transform=None):
        self.classes = list(self.classes)
        self.class_names = list(self.classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        binary_target = int(target) % 2
        return image, binary_target


