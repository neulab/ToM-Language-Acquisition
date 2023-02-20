import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from listener import Listener
from referential_game_env import ReferentialGameEnv

if __name__ == '__main__':
    listener = Listener(
        vocab_size=200,
        model_path=None
    ).cuda()

    envs = ReferentialGameEnv(
        max_len=20,
        eos_id=3,
        noop_penalty=0.5,
        length_penalty=0,
        batch_size=256,
        n_distr=2,
        game_file_path="game_file_hard_20.pt"
    )

    dev_envs = ReferentialGameEnv(
        max_len=20,
        eos_id=3,
        noop_penalty=0.5,
        length_penalty=0,
        batch_size=256,
        n_distr=2,
        game_file_path="game_file_dev.pt"
    )

    envs.game_file["sample_candidates"] = envs.game_file["similarity_rank"][:, :1000]
    dev_envs.game_file["sample_candidates"] = dev_envs.game_file["similarity_rank"][:, :1000]

    optimizer = torch.optim.Adam(listener.parameters())
    try: 
        for i in range(100000):
            obs = envs.reset()
            captions = envs.game_file["captions"][obs["images_ids"][range(256), obs["goal"]]]
            images = obs["images"]
            spk_lens = envs._find_eos(captions)
            images = torch.from_numpy(images).cuda()
            captions = captions.cuda()
            spk_lens = torch.LongTensor(spk_lens).cuda()
            pred_out, logits = listener.predict(images, captions, spk_lens, output_logp=True)
            loss = F.cross_entropy(logits, torch.from_numpy(obs["goal"]).cuda())
            accuracy = torch.mean((pred_out == torch.from_numpy(obs["goal"]).cuda()).float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 1000 == 999:
                print(loss.item(), accuracy.item())
                with torch.no_grad():
                    obs = dev_envs.reset()
                    captions = dev_envs.game_file["captions"][obs["images_ids"][range(256), obs["goal"]]]
                    images = obs["images"]
                    spk_lens = dev_envs._find_eos(captions)
                    images = torch.from_numpy(images).cuda()
                    captions = captions.cuda()
                    spk_lens = torch.LongTensor(spk_lens).cuda()
                    pred_out, logits = listener.predict(images, captions, spk_lens, output_logp=True)
                    loss = F.cross_entropy(logits, torch.from_numpy(obs["goal"]).cuda())
                    accuracy = torch.mean((pred_out == torch.from_numpy(obs["goal"]).cuda()).float())
                print(loss.item(), accuracy.item())
    except KeyboardInterrupt:
        torch.save(listener.state_dict(), "listener_200_hard_20.pt")
