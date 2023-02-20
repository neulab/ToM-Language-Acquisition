import torch
from torch import nn

a = torch.load("game_file_dev.pt")
images = torch.Tensor(a["images"])
cos = torch.mm(images, images.t()) \
    / torch.mm(
        torch.linalg.vector_norm(images, dim=1).unsqueeze(1),
        torch.linalg.vector_norm(images, dim=1).unsqueeze(0)
    )
cos_rank = torch.argsort(cos, descending=True)
a["similarity_rank"] = cos_rank
torch.save(a, "game_file_dev.pt")