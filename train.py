import os
import torch

from configs import config

from data.dataloader import get_train_loader
from models.mobilenet_light import MobileNetLight
from losses.arcface import ArcFace
from trainer.trainer import Trainer


def main():

    device = config.DEVICE

    print("Device:", device)

    train_loader = get_train_loader()

    print("Images :", len(train_loader.dataset))
    print("Classes:", train_loader.dataset.num_classes)

    model = MobileNetLight(
        embedding_size=config.EMBEDDING_SIZE
    ).to(device)

    criterion = ArcFace(
        embedding_size=config.EMBEDDING_SIZE,
        num_classes=train_loader.dataset.num_classes
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
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

    if os.path.exists(checkpoint_path):

        print("Loading checkpoint...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
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

        # ĐÃ SỬA
        loss = trainer.train_one_epoch(epoch)

        print(f"Average Loss: {loss:.4f}")

        checkpoint = {
            "epoch": epoch + 1,
            "batch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
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

        # log
        with open(log_path, "a") as f:
            f.write(
                f"Epoch {epoch+1}, Loss={loss:.4f}\n"
            )

        print("Checkpoint Saved")


if __name__ == "__main__":
    main()