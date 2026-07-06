import torch
import torch.nn.functional as F


class Evaluator:

    def __init__(self, model, device):

        self.model = model
        self.device = device

        self.model.eval()

    @torch.no_grad()
    def extract_embedding(self, image):

        image = image.unsqueeze(0).to(self.device)

        embedding = self.model(image)

        # Chuẩn hóa embedding
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding

    @torch.no_grad()
    def cosine_similarity(self, img1, img2):

        emb1 = self.extract_embedding(img1)
        emb2 = self.extract_embedding(img2)

        similarity = F.cosine_similarity(
            emb1,
            emb2,
            dim=1
        )

        return similarity.item()