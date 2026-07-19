import torch
from torchvision import transforms

from evaluation.lfw_dataset import LFWDataset
from evaluation.evaluator import Evaluator
from models.mobilenet_light import MobileNetLight


# =====================================
# Transform
# =====================================

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

# =====================================
# Dataset
# =====================================

dataset = LFWDataset(
    root_dir="datasets/lfw_funneled",
    pairs_file="datasets/lfw_funneled/pairs.txt",
    transform=transform
)

print("Total pairs:", len(dataset))

# =====================================
# Device
# =====================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =====================================
# Model
# =====================================

model = MobileNetLight(
    embedding_size=512
).to(device)

checkpoint = torch.load(
    "checkpoint/last_checkpoint.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# =====================================
# Evaluator
# =====================================

evaluator = Evaluator(
    model=model,
    device=device
)

# =====================================
# Tính similarity một lần
# =====================================

similarities = []
labels = []

print("Extracting similarities...")

for i in range(len(dataset)):

    img1, img2, label = dataset[i]

    sim = evaluator.cosine_similarity(
        img1,
        img2
    )

    similarities.append(sim)
    labels.append(label)

    if (i + 1) % 500 == 0:
        print(f"Processed {i+1}/{len(dataset)}")

# =====================================
# Search Threshold
# =====================================

best_acc = 0
best_threshold = 0

print("\nSearching threshold...\n")

for threshold in [i / 100 for i in range(0, 101)]:

    correct = 0

    for sim, label in zip(similarities, labels):

        pred = 1 if sim >= threshold else 0

        if pred == label:
            correct += 1

    acc = correct / len(labels)

    print(
        f"Threshold {threshold:.2f} "
        f"Accuracy {acc:.4f}"
    )

    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold

print("=" * 50)
print("Best Threshold :", best_threshold)
print("Best Accuracy  :", best_acc)
print("=" * 50)