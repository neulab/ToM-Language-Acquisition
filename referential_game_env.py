from random import sample
from typing import Dict, List, Tuple
import numpy as np
from listener import Listener
from math import log
from find_image import find_image_multi
from collections import Counter
import torch
import pdb

def H(n):
    #move somewhere else! calculates nth harmonic number
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)

class ReferentialGameEnv(object):
    def __init__(self, *,
                 max_len: int,
                 eos_id: int,
                 noop_penalty: float,
                 length_penalty: float,
                 batch_size: int,
                 n_distr: int,  # this might not be general
                 distribution: str = "uniform",
                 game_file_path: str = "game_file.pt",
                 captions_file: str,
                 **kwargs) -> None:
        super().__init__()
        self.max_len = max_len
        self.eos_id = eos_id
        self.noop_penalty = noop_penalty
        self.length_penalty = length_penalty
        self.batch_size = batch_size
        self.n_distr = n_distr
        self.distribution = distribution
        self.game_file_path = game_file_path
        self.image_size = (2048,)
        self.listener = Listener(**kwargs).cuda()
        self.target_ids = None
        self.captions_file_path = captions_file
        self.caption_length = 9
        self._get_game_file()

    def _get_game_file(self) -> np.array:
        import torch
        self.game_file = torch.load(self.game_file_path)
        for i in self.game_file:
            try:
                self.game_file[i] = torch.from_numpy(np.array(self.game_file[i]))
            except:
                pass

        filter_ids = []
        for idx, cap in enumerate(self.game_file["captions"]):
            if max(cap) < 199:
                filter_ids.append(idx)
        filter_id_set = set(filter_ids)
        filter_id_to_idx = {filter_ids[i]: i for i in range(len(filter_ids))}
        # pdb.set_trace()
        self.game_file["captions"] = self.game_file["captions"][filter_ids]
        self.game_file["images"] = self.game_file["images"][filter_ids]
        if "similarity_rank" not in self.game_file:
            self.game_file["sample_candidates"] = torch.from_numpy(
                np.random.randint(low=0,
                    high=len(filter_ids), size=(len(filter_ids), 100)
                )
            )
        else:
            sample_candidates = []
            import tqdm
            for i in tqdm.tqdm(filter_ids):
                tmp = []
                for j in self.game_file["similarity_rank"][i]:
                    j = j.item()
                    if j in filter_id_set and j != i:
                        tmp.append(filter_id_to_idx[j])
                    if len(tmp) == 1000:
                        break
                sample_candidates.append(tmp)
            self.game_file["sample_candidates"] = torch.LongTensor(sample_candidates)

        self.captions_file = torch.load(self.captions_file_path)
        self.captions_file = torch.from_numpy(np.array([[k[:self.caption_length] for k in j] for j in self.captions_file]))
        self.game_file["all_captions"] = self.captions_file[filter_ids]

    def _find_eos(self, actions: np.array) -> List[int]:
        eos_loc = [-1 for _ in range(len(actions))]
        for idx, i in enumerate(actions):
            for j in range(self.max_len):
                if i[j] == self.eos_id:
                    eos_loc[idx] = j+1
                    break
        return eos_loc

    def _new_game(self) -> np.array:
        import torch  # using old code here; change to numpy later
        sample_candidates = self.game_file["sample_candidates"]
        n = sample_candidates.size()[1]
        n_img = sample_candidates.size()[0]
        target_images = torch.randint(n_img, size=(self.batch_size,))
        if self.distribution == 'zipf':
            zipf_weights = np.array([1/(i*H(n)) for i in range(1, n+1)])
            distr_array = np.random.choice(n, (self.batch_size, self.n_distr+1), False, zipf_weights)
            distr_images = torch.from_numpy(distr_array)
        else:
            distr_images = torch.randint(n, size=(
            self.batch_size, self.n_distr + 1
        ))
        target_candidates = torch.index_select(
            sample_candidates, 0, target_images.view(-1)
        ).view(self.batch_size, n)
        distr_images = torch.gather(
            target_candidates, 1, distr_images).view(
                self.batch_size, self.n_distr+1
        )
        target_indices = torch.randint(
            self.n_distr + 1, size=(self.batch_size,))
        distr_images[range(self.batch_size), target_indices] \
            = target_images.view(self.batch_size)
        self.distr_images = distr_images.numpy()
        self.target_ids = target_indices.numpy()
        self.images = torch.index_select(
                self.game_file["images"], 0,
                distr_images.view(-1)
            ).view(self.batch_size, self.n_distr+1, *self.image_size).numpy()
        return dict(
            images=self.images,
            images_ids=distr_images.numpy(),
            goal=target_indices.numpy()
        )

    # def _render(self, actions, return_dict):
    #     import torch
    #     i2w = torch.load("i2w")
    #     for i in range(self.batch_size):
    #         print("goal\tchoice\timages")
    #         goal = [" " for i in range(self.n_distr+1)]
    #         goal[self.target_ids[i]] = "→"
    #         choice = [" " for i in range(self.n_distr+1)]
    #         choice[return_dict["choice"][i].item()] = "→"
    #         captions = torch.index_select(
    #             self.game_file["captions"], 0,
    #             torch.from_numpy(self.distr_images[i])
    #         ).cpu().tolist()
    #         captions = [
    #             ' '.join(i2w[i] for i in j) for j in captions
    #         ]
    #         for j in range(self.n_distr+1):
    #             print(f"{goal[j]}\t{choice[j]}\t{captions[j]}")
    #         print(f"sentence: {' '.join(i2w[j] for j in actions[i])}")

    def _render(self, actions, return_dict, name):
        import torch
        i2w = torch.load("i2w")
        for i in range(self.batch_size):
            with open(name + "_" + str(i) + ".html", "w") as html:
                goal = [" " for i in range(self.n_distr+1)]
                goal[self.target_ids[i]] = "→"
                choice = [" " for i in range(self.n_distr+1)]
                choice[return_dict["choice"][i].item()] = "→"
                captions = torch.index_select(
                    self.game_file["captions"], 0,
                    torch.from_numpy(self.distr_images[i])
                ).cpu().tolist()
                all_captions_torch = torch.index_select(
                    self.game_file["all_captions"], 0,
                    torch.from_numpy(self.distr_images[i])
                ).cpu().tolist()

                captions = [
                    ' '.join(i2w[i] for i in j) for j in captions
                ]
                all_captions = []
                for l in all_captions_torch:
                    tmp = []
                    for j in l:
                        caption_sentence = ' '.join(i2w[i] for i in j)
                        caption_no_tags = ' '.join(caption_sentence.split(' ')[1:-1])
                        tmp.append(caption_no_tags.split('<EOS>')[0])
                    all_captions.append(tmp)
                # write images + goal/choice status
                html.write("<html>\n<table>\n\t<tr>\n")
                for j in range(self.n_distr + 1):
                    header_str = "\t\t<th>Image " + str(j)
                    if goal[j] == "→":
                        header_str += " (Goal)"
                    if choice[j] == "→":
                        header_str += " (Choice)"
                    header_str += "</th>\n"
                    html.write(header_str)

                # write captions
                html.write("\t</tr>\n\t<tr>\n")
                for j in range(self.n_distr + 1):
                    html.write("\t\t<th>" + captions[j] + "</th>\n")
               
                # write images
                html.write("\t</tr>\n\t<tr>\n")
                for j in range(self.n_distr+1):
                    try:
                        img_link = find_image_multi(all_captions[j])
                        html.write("\t\t<td><img src='" + img_link + "'></td>\n")
                    except IndexError:
                        html.write("\t\t<td>IMAGE NOT FOUND</td>\n")

                # write output sentence
                html.write("\t</tr>\n\t<tr>\n")
                output = ' '.join(i2w[j] for j in actions[i])
                output = output.split('<EOS>')[0]
                html.write("\t\t<td colspan='" + str(self.n_distr+1) + f"'><center>output sentence: {output}</center></td>\n")
                # print(f"sentence: {' '.join(i2w[j] for j in actions[i])}")
                html.write("\t</tr>\n</table>\n</html>\n")

    def step(self, actions: np.array, render=False, name="None") -> Tuple[
        Dict[np.array, np.array], np.array
    ]:
        B = actions.shape[0]
        acc = []
        # for gold standard runs
        # import torch
        # actions = torch.index_select(self.game_file["captions"], 0, torch.from_numpy(self.distr_images[range(B), self.target_ids])).view(self.batch_size, -1).numpy()
        # actions = actions[:, :self.max_len]
        # listener act
        if True:
            action_len = self._find_eos(actions)
            for idx, i in enumerate(action_len):
                if i == -1:
                    action_len[idx] = self.max_len
            return_dict = self.listener.act(self.images, actions, action_len)
            if render:
                self._render(actions, return_dict, name)
        # reward calculation
        if True:
            reward = np.zeros_like(actions).astype(np.float32)
            eos_loc = self._find_eos(actions)
            # game reward
            for idx, i in enumerate(eos_loc):
                if (i != -1) and (return_dict["control"][idx] in [0, 1]):
                    if return_dict["choice"][idx] == self.target_ids[idx]:
                        reward[idx][i-1] = 1.0
                        acc.append(1)
                    else:
                        reward[idx][i-1] = -1.0
                        acc.append(0)
                else:
                    reward[idx][i-1] = -self.noop_penalty
            # length penalty
            for idx, i in enumerate(eos_loc):
                if i == -1:
                    reward[idx] -= self.length_penalty
                elif i < self.max_len - 1:
                    reward[idx][:i] -= self.length_penalty
        # observation
        if True:
            obs = dict()
            # feedback
            import torch
            # add choices and controls
            obs["choices"] = return_dict["choice"]
            obs["controls"] = return_dict["control"]
            obs["feedback"] = torch.index_select(
                self.game_file["captions"], 0,
                torch.from_numpy(self.distr_images[range(B), return_dict["choice"].cpu()])
            ).view(self.batch_size, -1).numpy()
            obs["feedback"] = obs["feedback"][:, :self.max_len]
            # obs["feedback_mask"] = (return_dict["control"] == 1).float()
            obs["ground_truth"] = torch.index_select(
                self.game_file["captions"], 0,
                torch.from_numpy(self.distr_images[range(B), self.target_ids])
            ).view(self.batch_size, -1).numpy()
            obs["ground_truth"] = obs["ground_truth"][:, :self.max_len]
            obs["accuracy"] = sum(acc) / len(acc) if len(acc) else 1/(self.n_distr + 1)
            # new game
            obs.update(self._new_game())

        return obs, reward

    def reset(self):
        return self._new_game()

    def close(self):
        pass

    def get_most_frequent_words(self, vocab_size) -> List[int]:
        return [tup[0] for tup in Counter(torch.flatten(self.game_file["captions"]).tolist()).most_common(vocab_size)]
