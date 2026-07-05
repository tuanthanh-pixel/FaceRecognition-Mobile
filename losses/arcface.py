import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    def __init__(
        self,
        embedding_size=512,
        num_classes=10575,
        scale=64.0,
        margin=0.5
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_size)
        )

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):

        # Chuẩn hóa embedding và weight
        embeddings = F.normalize(embeddings)
        weight = F.normalize(self.weight)

        # Cos(theta)
        cosine = F.linear(embeddings, weight)

        # Góc
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        # Cos(theta + m)
        target_logits = torch.cos(theta + self.margin)

        one_hot = F.one_hot(labels, self.num_classes).float()

        logits = cosine * (1 - one_hot) + target_logits * one_hot

        logits *= self.scale

        loss = F.cross_entropy(logits, labels)

        return loss, logits