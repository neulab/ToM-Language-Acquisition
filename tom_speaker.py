from tkinter import image_names
from typing import Tuple
import torch
from torch import nn
from coco_speaker import COCOSpeaker
from tom_listener import TOMListener
from speaker import Speaker
from listener import Listener
from beam_search import beam_search

class TOMSpeaker(nn.Module):
    '''
    Speaker with a ToM reranking module. Initialized from ppo.py.
    '''
    def __init__(self, maxlen, vocabsize, sigma, beam_size, tom_weight, use_coco=False, 
                word_list = None, use_pretrained = False, beam_search = False, loaded_model_paths = None):
        
        super(TOMSpeaker, self).__init__()
        self.beam_size = beam_size
        self.use_coco = use_coco
        if self.use_coco:
            self.speaker = COCOSpeaker(max_len=maxlen, vocabulary_size=vocabsize, D_img=2048, word_list=word_list)
            self.speaker = torch.jit.script(self.speaker)
        else:
            self.speaker = Speaker(max_len=maxlen, vocabulary_size=vocabsize)
        self.tom_listener = TOMListener(beam_size=beam_size, maxlen=maxlen, use_pretrained=use_pretrained)
        
        # load pretrained speaker if path is given
        if loaded_model_paths is not None:
            self.speaker.load_state_dict(torch.load(loaded_model_paths[0]))
            self.tom_listener.load_state_dict(torch.load(loaded_model_paths[1]))
            self.speaker.eval()
            
        self.tom_weight = tom_weight
        self.sigma = sigma
        self.beam_search = beam_search

    def sample(
        self, images: torch.FloatTensor, target_ids: torch.LongTensor, batch_size = 4,
        actions: torch.LongTensor = None, beam_size = None, include_pred = False
    ) -> Tuple[
        torch.LongTensor,  # actions (batch_size, max_len)
        torch.FloatTensor,  # logprobs (batch_size, max_len)
        torch.FloatTensor,  # entropy (batch_size, max_len)
        torch.FloatTensor,  # values (batch_size, max_len)
    ]:
        if beam_size is None:
            beam_size = self.beam_size
        B = images.size(0)
        target_images = images[range(B), target_ids]
        if self.use_coco:
            speaker_actions, speaker_logp, entropy, values = self.speaker.sample_multiple(target_images, actions, beam_size)
        elif self.beam_search:
            speaker_actions, speaker_logp, entropy, values = self.speaker.decode_with_beam_search(target_images, self.beam_size)
        else:
            speaker_actions, speaker_logp, entropy, values = self.speaker.sample_multiple(target_images, actions, beam_size)
       
        # reranking
        speaker_ranking = torch.log_softmax(torch.sum(speaker_logp, 2), dim=1)
        listener_ranking, pred = self.tom_listener._predict(images, target_ids, speaker_actions, beam_size = beam_size, include_pred = include_pred)
        ranking = speaker_ranking + self.tom_weight*listener_ranking
        
        # choosing best or random candidates
        best_candidate = torch.argmax(ranking, dim=0)
        random_candidate = torch.randint(0, speaker_actions.size(0), (best_candidate.shape[0],)).to(best_candidate.device)
        best_mask = (torch.bernoulli(torch.full(best_candidate.shape, self.sigma)).int()).to(best_candidate.device)
        rand_mask = (1 - best_mask).to(random_candidate.device)
        candidate = best_mask*best_candidate + rand_mask*random_candidate
        if include_pred:
            pred = pred[candidate, range(B)]
        
        # return action and value
        tom_action = speaker_actions[candidate,range(B),:]
        tom_logp = speaker_logp[candidate,range(B),:]
        tom_entropy = entropy[candidate,range(B),:]
        tom_values = values[candidate,range(B),:]
        if include_pred:
            return(tom_action, tom_logp, tom_entropy, tom_values, pred)
        else:
            return(tom_action, tom_logp, tom_entropy, tom_values)

    def supervised_loss(
        self,
        images: torch.FloatTensor,
        actions: torch.LongTensor,
        target_ids: torch.LongTensor,
        mask: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Compute loss for the supervised training of the model.
        """
        _, logprobs, _, _ = self.sample(images, target_ids, actions = actions, beam_size = 1)
        return -(logprobs.sum(-1) * mask.float()).mean()

    def update_tom_weight(self, new_weight):
        self.tom_weight = new_weight
    
    def update_sigma(self, new_sigma):
        self.sigma = new_sigma
