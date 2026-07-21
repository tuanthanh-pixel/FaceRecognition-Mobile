import os
import torch

from configs import config

from data.dataloader import get_train_loader
from models.mobilenet_light import MobileNetLight
from losses.arcface import ArcFace
from trainer.trainer import Trainer


# =====================================================
# In cấu hình thí nghiệm
# =====================================================

def print_config():

    print("=" * 60)
    print("Experiment Configuration")
    print("=" * 60)

    print(f"Model          : MobileNetLight")
    print(f"Embedding Size : {config.EMBEDDING_SIZE}")

    print(f"Margin         : {config.ARCFACE_MARGIN}")
    print(f"Scale          : {config.ARCFACE_SCALE}")

    print(f"Learning Rate  : {config.LEARNING_RATE}")
    print(f"Weight Decay   : {config.WEIGHT_DECAY}")

    print(f"Batch Size     : {config.BATCH_SIZE}")
    print(f"Epochs         : {config.EPOCHS}")

    print("=" * 60)


def main():

    device = config.DEVICE

    print("Device:", device)

    print_config()

    train_loader = get_train_loader()

    print("Images :", len(train_loader.dataset))
    print("Classes:", train_loader.dataset.num_classes)

    model = MobileNetLight(
        embedding_size=config.EMBEDDING_SIZE
    ).to(device)

    criterion = ArcFace(
        embedding_size=config.EMBEDDING_SIZE,
        num_classes=train_loader.dataset.num_classes,
        margin=config.ARCFACE_MARGIN,
        scale=config.ARCFACE_SCALE
    ).to(device)

    # ==========================
    # NEW : AdamW
    # ==========================

    optimizer = torch.optim.AdamW(
        list(model.parameters()) +
        list(criterion.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device
    )

    # =====================================
    # Tạo thư mục
    # =====================================

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR,
        "last_checkpoint.pth"
    )

    log_path = os.path.join(
        config.LOG_DIR,
        "train_log.txt"
    )

    # =====================================
    # Resume Training
    # =====================================

    start_epoch = 0
    start_batch = 0

    print("=" * 50)
    print("Checkpoint path:", checkpoint_path)
    print("Checkpoint exists:", os.path.exists(checkpoint_path))
    print("=" * 50)

    if os.path.exists(checkpoint_path):

        print("Loading checkpoint...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # ==========================
        # NEW
        # Tương thích checkpoint cũ
        # ==========================

        if "criterion_state_dict" in checkpoint:

            criterion.load_state_dict(
                checkpoint["criterion_state_dict"]
            )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"]

        start_batch = checkpoint.get("batch", 0)

        print(f"Resume from Epoch {start_epoch}")
        print(f"Last saved Batch : {start_batch}")

    # =====================================
    # Training Loop
    # =====================================

    for epoch in range(start_epoch, config.EPOCHS):

        print("=" * 50)
        print(f"Epoch {epoch+1}/{config.EPOCHS}")

        loss = trainer.train_one_epoch(epoch)

        print(f"Average Loss: {loss:.4f}")

        checkpoint = {

            # ==========================
            # NEW
            # ==========================
            "checkpoint_version": 2,

            "epoch": epoch + 1,
            "batch": 0,

            "model_state_dict": model.state_dict(),

            # ==========================
            # NEW
            # ==========================
            "criterion_state_dict": criterion.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "loss": loss,

            # Hyperparameters
            "margin": config.ARCFACE_MARGIN,
            "scale": config.ARCFACE_SCALE,
            "learning_rate": config.LEARNING_RATE,
            "weight_decay": config.WEIGHT_DECAY,
            "batch_size": config.BATCH_SIZE,
            "embedding_size": config.EMBEDDING_SIZE

        }

        # checkpoint mới nhất
        torch.save(
            checkpoint,
            checkpoint_path
        )

        # checkpoint từng epoch
        torch.save(
            checkpoint,
            os.path.join(
                config.CHECKPOINT_DIR,
                f"epoch_{epoch+1}.pth"
            )
        )

        # =====================================
        # Log
        # =====================================

        with open(log_path, "a") as f:

            if epoch == start_epoch:

                f.write("\n")
                f.write("=" * 60 + "\n")
                f.write("Experiment Configuration\n")
                f.write("=" * 60 + "\n")

                f.write(f"Margin        : {config.ARCFACE_MARGIN}\n")
                f.write(f"Scale         : {config.ARCFACE_SCALE}\n")
                f.write(f"Learning Rate : {config.LEARNING_RATE}\n")
                f.write(f"Weight Decay  : {config.WEIGHT_DECAY}\n")
                f.write(f"Batch Size    : {config.BATCH_SIZE}\n")
                f.write(f"Epochs        : {config.EPOCHS}\n")

                f.write("\n")

            f.write(
                f"Epoch {epoch+1} | Loss = {loss:.4f}\n"
            )

        print("Checkpoint Saved")

    print("=" * 60)
    print("Training Finished")


if __name__ == "__main__":
    main()