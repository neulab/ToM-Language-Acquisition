from typing import List
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
import random
import numpy as np


class Beholder(nn.Module):
    def __init__(self):
        super(Beholder, self).__init__()
        D_img = 2048
        D_hid = 512
        dropout = 0.5
        self.img_to_hid = nn.Linear(D_img, D_hid)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, img):
        h_img = img
        h_img = self.img_to_hid(h_img)
        h_img = self.drop(h_img)
        return h_img


class Listener(nn.Module):

    def __init__(
        self,
        beholder=None,
        D_hid=512,
        D_emb=256,
        vocab_size=200,
        dropout=0.5,
        theta_1=0.4,
        theta_2=0.9,
        maxlen=20,
        model_path=None,
        load_checkpoint=True
    ):
        super(Listener, self).__init__()
        self.rnn = nn.GRU(D_emb, D_hid, 1, batch_first=True)
        self.emb = nn.Linear(vocab_size, D_emb)
        self.hid_to_hid = nn.Linear(D_hid, D_hid)
        self.drop = nn.Dropout(p=dropout)
        self.D_hid = D_hid
        self.D_emb = D_emb
        self.vocab_size = vocab_size
        if beholder is None:
            self.beholder = Beholder()
        else:
            self.beholder = beholder
        # self.loss_fn = nn.CrossEntropyLoss(reduction='none').to(device=args.device)
        self.optimizer = optim.Adam(self.parameters())
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.maxlen = maxlen
        self.stop = torch.Tensor([0]).long()
        self._reset()
        
        if load_checkpoint:
            if model_path is None:
                model_path = f"listener_{vocab_size}_{maxlen}.pt"
            if model_path is not None:
                self.load_state_dict(torch.load(model_path))

    def _reset(self):
        self.ground_truth_queue = None

    def forward(self, spk_msg, spk_msg_lens, num_beams=1, multi=False):
        # batch_size = spk_msg.shape[0]
        batch_size = spk_msg.shape[-2]
        if multi:
            spk_msg = spk_msg.flatten(start_dim=0, end_dim=1)
            spk_msg_lens = spk_msg_lens.flatten(start_dim=0, end_dim=1)
        device = next(self.parameters()).device
        h_0 = torch.zeros(1, batch_size*num_beams, self.D_hid, device=device)

        if spk_msg.type() in ['torch.FloatTensor', 'torch.cuda.FloatTensor']:
            spk_msg_emb = self.emb(spk_msg.float())
        elif spk_msg.type() in ['torch.LongTensor', 'torch.cuda.LongTensor']:
            spk_msg[spk_msg >= self.vocab_size] = 1  # <unk>
            spk_msg_emb = F.embedding(
                spk_msg.clone(), self.emb.weight.transpose(0, 1))
            spk_msg_emb += self.emb.bias
        else:
            print(spk_msg.type())
            raise NotImplementedError
        spk_msg_emb = self.drop(spk_msg_emb)
        try:
            pack = nn.utils.rnn.pack_padded_sequence(
                spk_msg_emb, spk_msg_lens, batch_first=True, enforce_sorted=False)
        except:
            import pdb
            pdb.set_trace()
        self.rnn.flatten_parameters()
        _, h_n = self.rnn(pack, h_0)
        h_n = h_n[-1:, :, :]
        out = h_n.transpose(0, 1).view(num_beams*batch_size, self.D_hid)
        out = self.hid_to_hid(out)
        return out

    def get_loss_acc(self, image, distractor_images, spk_msg, spk_msg_lens,
                     reduction='mean', shuffle=True, output_pred=False,
                     output_logits=False):
        batch_size = spk_msg.shape[0]

        if reduction != 'none':
            spk_msg_lens, sorted_indices = torch.sort(
                spk_msg_lens, descending=True)
            spk_msg = spk_msg.index_select(0, sorted_indices)
            image = image.index_select(0, sorted_indices)

        h_pred = self.forward(spk_msg, spk_msg_lens.cpu())
        h_pred = h_pred.unsqueeze(1).repeat(1, 1 + len(distractor_images), 1)

        all_images = len(distractor_images) + 1
        img_idx = [list(range(all_images)) for _ in range(batch_size)]
        for c in img_idx:
            if shuffle:
                random.shuffle(c)

        target_idx = torch.tensor(
            np.argmax(np.array(img_idx) == 0, -1), dtype=torch.long, device=device)

        h_img = [self.beholder(image)] + [self.beholder(img)
                                          for img in distractor_images]
        h_img = torch.stack(h_img, dim=0).permute(1, 0, 2)
        for i in range(batch_size):
            h_img[i] = h_img[i, img_idx[i], :]

        logits = 1 / torch.mean(torch.pow(h_pred - h_img, 2),
                                2).view(-1, 1 + len(distractor_images))

        pred_outs = torch.argmax(logits, dim=-1).cpu().numpy()
        batch_inds = target_idx.cpu().numpy()

        acc = np.mean(np.equal(batch_inds, pred_outs))
        loss = F.cross_entropy(logits, target_idx, reduction=reduction)
        if not output_pred:
            if not output_logits:
                return loss, acc
            else:
                return loss, acc, logits
        else:
            if not output_logits:
                return loss, acc, pred_outs
            else:
                return loss, acc, pred_outs, logits

    def predict(self, images, spk_msg, spk_msg_lens, num_beams=1, multi=False, output_logp=False):
        h_pred = self.forward(spk_msg, spk_msg_lens.cpu(), num_beams=num_beams, multi=multi)
        h_pred = h_pred.unsqueeze(1).repeat(1, images.size()[1], 1)
        h_img = self.beholder(images)
        if multi:
            h_pred = h_pred.reshape(num_beams, images.size()[0], 3, -1)
            h_img = h_img.unsqueeze(0).repeat(num_beams, 1, 1, 1)
            logits = 1 / torch.mean(torch.pow(h_pred - h_img, 2), 3).view(num_beams,-1,images.size()[1])
        else:
            logits = 1 / torch.mean(torch.pow(h_pred - h_img, 2), 2).view(-1,images.size()[1])
        pred_outs = torch.argmax(logits, dim=-1)
        if output_logp:
            return pred_outs, torch.log_softmax(logits, dim=-1)
        else:
            return pred_outs

    def test(self, image, distractor_images, spk_msg, spk_msg_lens):
        self.eval()
        loss, acc = self.get_loss_acc(
            image, distractor_images, spk_msg, spk_msg_lens)
        return loss.detach().cpu().numpy(), acc

    def _pad(self, x: torch.Tensor):
        """
        x: L x B
        """
        L = x.size()[0]
        if L >= self.maxlen:
            return x[:self.maxlen]
        else:
            return torch.cat([x, torch.zeros(self.maxlen-L, x.size()[1]).to(x.device)], dim=0)

    def act(self, world: torch.Tensor, sentence: torch.Tensor, sentence_len: torch.LongTensor):
        sentence_len = torch.Tensor(sentence_len).to(next(self.parameters()).device)
        sentence = torch.from_numpy(sentence).to(next(self.parameters()).device)
        world = torch.from_numpy(world).to(next(self.parameters()).device)
        _, logp = self.predict(world, sentence, sentence_len, output_logp=True)
        prob = logp.softmax(dim=-1)

        max_prob, max_idx = prob.max(dim=-1)

        # Case 1: high confidence
        mask_1 = (max_prob >= self.theta_2).float()
        # Case 2: mid confidence
        mask_2 = ((self.theta_1 <= max_prob) & (
            max_prob < self.theta_2)).float()
        # Case 3: low confidence
        mask_3 = 1 - mask_1 - mask_2

        return_dict = dict(
            control=(mask_2 + mask_3 * 2).int(),
            choice=max_idx
        )

        return return_dict
