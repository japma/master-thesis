from torch.utils.data import Dataset
from torchvision import datasets


class BinaryMNISTDataset(Dataset):
    classes = ("even", "odd")

    def __init__(
        self, root, train: bool = True, download: bool = True, transform=None
    ) -> None:
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

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        binary_target = int(target) % 2
        return image, binary_target
