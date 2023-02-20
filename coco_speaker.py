from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
import torchvision.models as models
from PIL import Image
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from beam_search import beam_search


def ppo_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a layer with an orthogonal matrix and a bias vector. This is
    for PPO
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class COCOSpeaker(nn.Module):
    class _block(nn.Module):
        def __init__(self, hid):
            super().__init__()
            self.hid = hid
            self.dense1 = nn.Sequential(
                nn.Linear(hid, hid),
                nn.GELU(),
                nn.Linear(hid, hid),
            )
            self.dense2 = nn.Sequential(
                nn.Linear(hid, hid),
                nn.GELU(),
                nn.Linear(hid, hid),
            )

        def forward(self, x):
            y = nn.functional.layer_norm(x, [self.hid])
            y = self.dense1(y)
            y = nn.functional.layer_norm(y + x, [self.hid])
            return x + self.dense2(y)

    def __init__(
        self, *, vocabulary_size: int, max_len: int, D_img: int, word_list: List[int]
    ):
        super(COCOSpeaker, self).__init__()

        self.feature_resizer = nn.Linear(2304, D_img)

        self.max_len = max_len
        # image_encoder = nn.Sequential(
        # *list(models.resnet18(pretrained=True).children())[:-1]
        # )
        # Dry run to get the output size
        # out = image_encoder(torch.zeros(1, 3, 224, 224))
        # out_size = out.view(-1).size()[0]
        out_size = D_img
        # Add feature resizer the decoder
        self.encoder = nn.Sequential(
            COCOSpeaker._block(out_size),
            nn.LayerNorm(out_size),
            nn.Linear(out_size, 512),
        )
        self.word_emb = nn.Embedding(vocabulary_size, 512)
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
        self.word_mask = torch.ones(vocabulary_size) * (-1000)
        for i in word_list:
            self.word_mask[i] = 0
        self.word_mask = nn.Parameter(self.word_mask, requires_grad=False)

    @torch.jit.export
    def get_action_and_value(
        self, images: torch.FloatTensor, actions: Optional[torch.LongTensor] = None, ) -> Tuple[
        torch.LongTensor,  # actions (batch_size, max_len)
        torch.FloatTensor,  # logprobs (batch_size, max_len)
        torch.FloatTensor,  # entropy (batch_size, max_len)
        torch.FloatTensor,  # values (batch_size, max_len)
    ]:
        B = images.size(0)
        image_representation = self.encoder(images)
        ix = torch.zeros_like(image_representation)
        action_list = []
        logprobs = []
        entropies = []
        values = []
        hx = image_representation
        for i in range(self.max_len):
            hx = self.decoder(hx, ix)
            logits = self.actor(hx)
            logits += self.word_mask
            logits = torch.log_softmax(logits, dim=-1)  # normalize
            if actions is not None:
                action = actions[:, i]
            else:
                action = torch.multinomial(torch.exp(logits), 1).squeeze(-1)
            logprob = torch.gather(logits, dim=-1, index=action.unsqueeze(-1)).squeeze(
                -1
            )
            entropy = -torch.sum(torch.softmax(logits, dim=-1) * logits, dim=-1)
            value = self.critic(hx)
            ix = self.word_emb(action)

            action_list.append(action)
            logprobs.append(logprob)
            entropies.append(entropy)
            values.append(value)

        action_list = torch.stack(action_list, dim=1)
        logprobs = torch.stack(logprobs, dim=1)
        entropies = torch.stack(entropies, dim=1)
        values = torch.stack(values, dim=1)
        return action_list, logprobs, entropies, values

    @torch.jit.export
    def supervised_loss(
        self,
        images: torch.FloatTensor,
        actions: torch.LongTensor,
        # bboxes: torch.FloatTensor,
        mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute the loss for the supervised training of the model.
        """
        _, logprobs, _, _ = self.get_action_and_value(images=images, actions = actions)
        # _, logprobs, _, _ = self(images, bboxes, actions)
        logprobs = torch.where(logprobs < -1000, torch.zeros_like(logprobs), logprobs)
        # logprobs[actions==1] = 0
        return -(logprobs.sum(-1) * mask.float()).mean()

    @torch.jit.export
    def get_image_representation(
        self, image_feature: torch.FloatTensor, box_feature: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Forward pass for the COCO Speaker model.
        """
        return self.feature_resizer(torch.cat([image_feature, box_feature], dim=1))

    @torch.jit.export
    def forward(
        self,
        image_feature: torch.FloatTensor,
        box_feature: torch.FloatTensor,
        actions: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        torch.LongTensor,  # actions (batch_size, max_len)
        torch.FloatTensor,  # logprobs (batch_size, max_len)
        torch.FloatTensor,  # entropy (batch_size, max_len)
        torch.FloatTensor,  # values (batch_size, max_len)
    ]:
        return self.get_action_and_value(
            images=self.get_image_representation(image_feature, box_feature), actions=actions,
        )

    def step(self, hid, input):
        ix = self.word_emb(input)
        if input[0] == 0: # RobertaTokenizer.bos_token_id
            ix = torch.zeros_like(hid)
        hx = self.decoder(hid, ix)
        logits = self.actor(hx)
        logits += self.word_mask
        logits = torch.log_softmax(logits, dim=-1)
        return logits, hx
    
    def decode_with_beam_search(
        self,
        images: torch.FloatTensor,
        beam_size: int = 5
    ):
        """
        Decode the images with beam search.
        """
        B = images.size(0)
        image_representation = self.encoder(images)
        def dummy_generator(image_representation):
            yield image_representation
        return beam_search(
            self.step,
            beam_size=beam_size,
            max_len=self.max_len,
            eos_id=2,
            bos_id=1,
            dataloader=dummy_generator(image_representation)
        )
    @torch.jit.export
    def sample_multiple(
            self,
            images: torch.FloatTensor,
            actions: Optional[torch.LongTensor] = None,
            beam_size: int = 5,
            batch_size: int = 4
        ) -> Tuple[
            torch.LongTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor]:
            """
            Sample multiple images from the speaker's distribution.
            """
            if actions is None:
                images = images.repeat(beam_size,1)
            actions, logprobs, entropy, values = self.get_action_and_value(images=images, actions=actions,)
            return(torch.reshape(actions, (beam_size, -1, self.max_len)), 
            torch.reshape(logprobs, (beam_size, -1, self.max_len)),
            torch.reshape(entropy, (beam_size, -1, self.max_len)),
            torch.reshape(values, (beam_size, -1, self.max_len)))
