import torch
from torch import nn
from listener import Listener

class TOMListener(Listener):
    '''
    Trainable listener module that speaker can use to learn listener behavior.
    Initialized from tom_speaker.py.
    '''
    def __init__(self, sigma=0.5, beam_size=1, maxlen=10, use_pretrained = False):
        super(TOMListener, self).__init__(maxlen=maxlen, load_checkpoint=use_pretrained)
        self.sigma = sigma
        self.beam_size = beam_size
        self.eos_id = 3

    # compute lengths of candidate utterances
    def candidate_lengths_2d(self, candidates):
        eos_loc = [len(candidates[0]) for _ in range(len(candidates))]
        for idx, i in enumerate(candidates):
            for j in range(len(i)):
                if i[j] == self.eos_id:
                    eos_loc[idx] = j+1
        return(eos_loc)

    def candidate_lengths(self, candidates):
        num_beams = self.beam_size
        candidates = (candidates.cpu()).numpy()
        if candidates.ndim == 3:
            all_eos_loc = []
            for idx, i in enumerate(candidates):
                all_eos_loc.append(self.candidate_lengths_2d(i))
            return(torch.FloatTensor(all_eos_loc))
        else:
            return(torch.FloatTensor(self.candidate_lengths_2d(candidates)))

    # generate predicted logprobs for target image given all candidate utterances
    # used by tom_speaker to rerank utterances
    def _predict(self, images, target_ids, candidates: torch.LongTensor, multi = True, beam_size = None, include_pred = False) -> torch.FloatTensor:
        """
        Args:
            images: (batch_size, D_img)
            candidates: (N_candidates, batch_size, max_len)
        Returns:
            logprobs: (batch_size, max_len)
        """
        if beam_size is None:
            beam_size = self.beam_size
        N_candidates, batch_size, max_len = candidates.size()
        candidate_length = self.candidate_lengths(candidates)
        with torch.no_grad():
            pred, logp = self.predict(images, candidates, candidate_length, 
                num_beams = beam_size, multi = multi, output_logp=True)
        logp = logp[:, range(batch_size), target_ids]
        if include_pred:
            return(logp, pred)
        else:
            return(logp, None)

    # loss for retraining to match actual listener
    def supervised_loss(self, images, target_ids, sentences, mask):
        sentence_length = self.candidate_lengths(sentences)
        _, logp = self.predict(images, sentences, sentence_length, output_logp=True)
        logprobs = -logp[range(images.size(0)), target_ids]
        return (logprobs * mask.float()).mean()

    # for training separately to rest of network
    def train_step(self, images, target_ids, sentences, mask):
        self.train(True)
        loss = self.supervised_loss(images, target_ids, sentences, mask)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train(False)
        return(loss)
        
