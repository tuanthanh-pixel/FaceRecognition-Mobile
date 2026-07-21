import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from configs import config


class Trainer:

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_loader,
        device
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device

    def train_one_epoch(self, epoch):

        self.model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch+1}",
            leave=True
        )

        for batch_idx, batch in progress_bar:

            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # ==================================
            # Forward
            # ==================================

            embeddings = self.model(images)

            logits = self.criterion(
                embeddings,
                labels
            )

            loss = F.cross_entropy(
                logits,
                labels
            )

            # ==================================
            # Backward
            # ==================================

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

            # ==================================
            # Save checkpoint mỗi 500 batch
            # ==================================

            if (batch_idx + 1) % 500 == 0:

                checkpoint = {
                    "epoch": epoch,
                    "batch": batch_idx + 1,
                    "model_state_dict": self.model.state_dict(),
                    "criterion_state_dict": self.criterion.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss.item(),
                }

                torch.save(
                    checkpoint,
                    os.path.join(
                        config.CHECKPOINT_DIR,
                        "last_checkpoint.pth"
                    )
                )

                print(f"\nCheckpoint saved at Batch {batch_idx + 1}")

        return running_loss / len(self.train_loader)