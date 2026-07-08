import torch
from torchvision import transforms

from evaluation.lfw_dataset import LFWDataset
from evaluation.evaluator import Evaluator
from evaluation.metrics import accuracy

from models.mobilenet_light import MobileNetLight


# =====================================================
# Transform
# =====================================================

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

# =====================================================
# Dataset
# =====================================================

dataset = LFWDataset(
    root_dir="datasets/lfw_funneled",
    pairs_file="datasets/lfw_funneled/pairs.txt",
    transform=transform
)

print("Total pairs:", len(dataset))

# =====================================================
# Device
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =====================================================
# Model
# =====================================================

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
# =====================================================
# Evaluator
# =====================================================

evaluator = Evaluator(
    model=model,
    device=device
)

# =====================================================
# Evaluation
# =====================================================

correct = 0
total = len(dataset)

threshold = 0.5

positive_similarity = []
negative_similarity = []

for i in range(total):

    img1, img2, label = dataset[i]

    similarity = evaluator.cosine_similarity(
        img1,
        img2
    )

    # Lưu similarity để thống kê
    if label == 1:
        positive_similarity.append(similarity)
    else:
        negative_similarity.append(similarity)

    prediction = 1 if similarity >= threshold else 0

    if prediction == label:
        correct += 1

    # Hiển thị tiến độ
    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{total}")

# =====================================================
# Result
# =====================================================

acc = accuracy(correct, total)

print("=" * 50)
print(f"Correct : {correct}")
print(f"Wrong   : {total - correct}")
print(f"Total   : {total}")
print(f"Accuracy: {acc:.4f}")

print("=" * 50)
print("Positive Similarity Statistics")
print(f"Mean : {sum(positive_similarity)/len(positive_similarity):.4f}")
print(f"Min  : {min(positive_similarity):.4f}")
print(f"Max  : {max(positive_similarity):.4f}")

print("=" * 50)
print("Negative Similarity Statistics")
print(f"Mean : {sum(negative_similarity)/len(negative_similarity):.4f}")
print(f"Min  : {min(negative_similarity):.4f}")
print(f"Max  : {max(negative_similarity):.4f}")