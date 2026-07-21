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

        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_size)
        )

        nn.init.xavier_uniform_(self.weight)

        self.update_margin_params(margin)

    def update_margin_params(self, margin):

        self.margin = margin

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):

        embeddings = F.normalize(embeddings)
        weight = F.normalize(self.weight)

        cosine = F.linear(embeddings, weight)

        cosine = torch.clamp(
            cosine,
            -1.0 + 1e-7,
            1.0 - 1e-7
        )

        sine = torch.sqrt(
            1.0 - cosine.pow(2)
        )

        target_logits = (
            cosine * self.cos_m
            -
            sine * self.sin_m
        )

        target_logits = torch.where(
            cosine > self.th,
            target_logits,
            cosine - self.mm
        )

        one_hot = F.one_hot(
            labels,
            self.weight.size(0)
        ).float()

        logits = (
            cosine * (1 - one_hot)
            +
            target_logits * one_hot
        )

        logits *= self.scale

        return logits