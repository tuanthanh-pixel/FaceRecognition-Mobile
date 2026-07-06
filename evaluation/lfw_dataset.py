import os

from PIL import Image
from torch.utils.data import Dataset


class LFWDataset(Dataset):

    def __init__(
        self,
        root_dir,
        pairs_file,
        transform=None
    ):

        self.root_dir = root_dir
        self.pairs_file = pairs_file
        self.transform = transform
        self.pairs = []

        self._load_pairs()

    def _load_pairs(self):

        with open(self.pairs_file, "r") as f:
            lines = f.readlines()

        # Bỏ dòng đầu tiên: "10 300"
        lines = lines[1:]

        for line in lines:

            items = line.strip().split()

            # Positive Pair (cùng người)
            if len(items) == 3:

                name = items[0]

                img1 = os.path.join(
                    self.root_dir,
                    name,
                    f"{name}_{int(items[1]):04d}.jpg"
                )

                img2 = os.path.join(
                    self.root_dir,
                    name,
                    f"{name}_{int(items[2]):04d}.jpg"
                )

                label = 1

            # Negative Pair (khác người)
            elif len(items) == 4:

                name1 = items[0]
                name2 = items[2]

                img1 = os.path.join(
                    self.root_dir,
                    name1,
                    f"{name1}_{int(items[1]):04d}.jpg"
                )

                img2 = os.path.join(
                    self.root_dir,
                    name2,
                    f"{name2}_{int(items[3]):04d}.jpg"
                )

                label = 0

            else:
                continue

            self.pairs.append(
                (img1, img2, label)
            )

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, index):

        img1_path, img2_path, label = self.pairs[index]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform is not None:

            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label