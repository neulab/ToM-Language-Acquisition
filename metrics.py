from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import torch
from appdirs import user_cache_dir
from pathlib import Path
from collections import defaultdict
import numpy as np
import spacy_sentence_bert as spacy
import re

def remove_tag(text):
    return re.sub(r'<.*?>', '', text)

def sentence_length(text):
    text_list = text.strip().split()
    ignore_list = ['<BOS>', '<UNK>']
    i = 0
    for word in text_list:
        if word == '<EOS>':
            return i
        if word not in ignore_list:
            i += 1
    return i

def num_nouns(model, text):
    length = sentence_length(text)
    text = " ".join(text.strip().split()[:length])
    doc = model(text)
    pos = [token.pos_ for token in doc]
    return(pos.count('NOUN'))

class Fluency(object):
    def __init__(self, device="cuda") -> None:
        super().__init__()
        self.device = device
        model_id = "gpt2-large"
        self.model = GPT2LMHeadModel.from_pretrained("gpt2_coco")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.unigram_logprob = self._get_unigram_prob()

    def _get_unigram_prob(self) -> None:
        cachedir = Path(user_cache_dir("interactgym"))
        if Path(cachedir / "unigram_logfreq.pt").exists():
            unigram_logprob = torch.load(cachedir / "unigram_logfreq.pt")
            unigram_logprob.default_factory = lambda: unigram_logprob["<unk>"]
            return unigram_logprob
        else:
            if not Path(cachedir).exists():
                Path(cachedir).mkdir()

            train = load_dataset("wikitext", "wikitext-2-raw-v1",
                                 split="train")
            encodings = self.tokenizer("\n\n".join(train["text"]),
                                       return_tensors="pt")
            freq = defaultdict(float)
            input_ids = encodings.input_ids.view(-1)
            print(input_ids.size())
            for idx, i in enumerate(torch.bincount(input_ids)):
                freq[idx] = np.log(i+1) - np.log(input_ids.size(0))
            freq["<unk>"] = -np.log(input_ids.size(0))
            torch.save(freq, cachedir / "unigram_logfreq.pt")
            freq.default_factory = lambda: -np.log(input_ids.size(0))
            return freq

    def _get_score(self, text: str) -> float:
        encodings = self.tokenizer(text + ".", return_tensors="pt")
        input_ids = encodings.input_ids.view(-1)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            unigram_log_likelihood = sum(
                map(
                    lambda x: self.unigram_logprob[x],
                    input_ids.cpu().view(-1).tolist()
                )
            )

            # SLOR = (log(P_M(x)) - log(P_unigram(x))) / |x|
            # https://aclanthology.org/K18-1031.pdf
            slor = - outputs[0].item() \
                - unigram_log_likelihood / input_ids.size(0)

        return slor
    
    def __call__(self, text: str) -> float:
        return self._get_score(
            remove_tag(text)  # remove tags
        )


class SemanticSimilarity(object):
    def __init__(self) -> None:
        super().__init__()
        self.nlp = spacy.load_model('en_stsb_roberta_large')
    
    def _get_score(self, text: str, target: str) -> float:
        doc1 = self.nlp(text)
        doc2 = self.nlp(target)
        return doc1.similarity(doc2)

    def __call__(self, text: str, target: str) -> float:
        return self._get_score(
            remove_tag(text),  # remove tags
            remove_tag(target)
        )


if __name__ == "__main__":
    fluency = Fluency(device="cpu")
    print(fluency("The quick brown fox jumps over the lazy dog."))
    semantic_similarity = SemanticSimilarity()
    print(semantic_similarity("The quick brown fox jumps over the lazy dog.", "The quick brown dog jump over the lazy fox."))
