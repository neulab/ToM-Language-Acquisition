from typing import Tuple
import torch
from torch import nn
import numpy as np
import torchvision.models as models


def ppo_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a layer with an orthogonal matrix and a bias vector. This is
    for PPO
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Speaker(nn.Module):
    def __init__(self, *, vocabulary_size: int, max_len: int):
        super(Speaker, self).__init__()
        self.max_len = max_len
        # image_encoder = nn.Sequential(
            # *list(models.resnet18(pretrained=True).children())[:-1]
        # )
        # Dry run to get the output size
        # out = image_encoder(torch.zeros(1, 3, 224, 224))
        # out_size = out.view(-1).size()[0]
        out_size = 2048
        # Add feature resizer the decoder
        self.encoder = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.decoder = nn.GRUCell(
            input_size=512,
            hidden_size=512,
        )
        self.critic = nn.Sequential(
            ppo_layer_init(nn.Linear(512, 64)),
            nn.Tanh(),
            ppo_layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            ppo_layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            ppo_layer_init(nn.Linear(512, 64)),
            nn.Tanh(),
            ppo_layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            ppo_layer_init(nn.Linear(64, vocabulary_size), std=0.01),
        )

    def get_action_and_value(
        self,
        images: torch.FloatTensor,
        actions: torch.LongTensor = None,
        targets: torch.LongTensor = None
    ) -> Tuple[
        torch.LongTensor,  # actions (batch_size, max_len)
        torch.FloatTensor,  # logprobs (batch_size, max_len)
        torch.FloatTensor,  # entropy (batch_size, max_len)
        torch.FloatTensor,  # values (batch_size, max_len)
    ]:
        image_representation = self.encoder(images)
        hx = torch.zeros_like(image_representation)
        action_list = []
        logprobs = []
        entropies = []
        values = []
        for i in range(self.max_len):
            hx = self.decoder(
                image_representation, hx
            )
            logits = self.actor(hx)
            dist = torch.distributions.Categorical(logits=logits)
            if actions is not None:
                action = actions[:, i]
            else:
                action = dist.sample()
            logprob = dist.log_prob(action)
            entropy = dist.entropy()
            value = self.critic(hx)
            action_list.append(action)
            logprobs.append(logprob)
            entropies.append(entropy)
            values.append(value)

        action_list = torch.stack(action_list, dim=1)
        logprobs = torch.stack(logprobs, dim=1)
        entropies = torch.stack(entropies, dim=1)
        values = torch.stack(values, dim=1)
        return action_list, logprobs, entropies, values

    def supervised_loss(
        self,
        images: torch.FloatTensor,
        actions: torch.LongTensor,
        mask: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Compute the loss for the supervised training of the model.
        """
        actions[actions>=200] = 1
        _, logprobs, _, _ = self.get_action_and_value(images, actions)
        # logprobs[actions==1] = 0
        return -(logprobs.sum(-1) * mask.float()).mean()

    def sample_multiple(
            self,
            images: torch.FloatTensor,
            actions: torch.LongTensor = None,
            beam_size: int = 5,
            batch_size: int = 4
        ):
            """
            Sample multiple images from the speaker's distribution.
            """
            if actions is None:
                images = images.repeat(beam_size,1)
            actions, logprobs, entropy, values = self.get_action_and_value(images, actions)
            return(torch.reshape(actions, (beam_size, -1, self.max_len)), 
            torch.reshape(logprobs, (beam_size, -1, self.max_len)),
            torch.reshape(entropy, (beam_size, -1, self.max_len)),
            torch.reshape(values, (beam_size, -1, self.max_len)))