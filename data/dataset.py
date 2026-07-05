import os

from PIL import Image

from torch.utils.data import Dataset


class CASIADataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir

        self.transform = transform

        self.samples = []

        self.class_to_idx = {}

        self._load_dataset()

        # Số lượng class trong dataset
        self.num_classes = len(self.class_to_idx)

    def _load_dataset(self):

        identities = sorted(os.listdir(self.root_dir))

        for label, identity in enumerate(identities):

            identity_path = os.path.join(
                self.root_dir,
                identity
            )

            if not os.path.isdir(identity_path):
                continue

            self.class_to_idx[identity] = label

            for image_name in os.listdir(identity_path):

                image_path = os.path.join(
                    identity_path,
                    image_name
                )

                self.samples.append(
                    (
                        image_path,
                        label
                    )
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "path": image_path
        }