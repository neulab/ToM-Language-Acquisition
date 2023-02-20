from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from nltk.translate import bleu
import nltk.translate.bleu_score as bs
import torch
import argparse

def sentence_length(sent):
	length = 0
	for word in sent:
		if word == '<EOS>' or word == 3:
			return length
		length += 1
	return length

def compute_bleu(pretrained_path):
        i2w = torch.load('i2w')
        actions = torch.load('wandb/' + pretrained_path + '/files/all_actions.pt', map_location = torch.device('cpu'))
        feedback = torch.load('wandb/' + pretrained_path + '/files/all_feedback.pt', map_location = torch.device('cpu'))
        actions = torch.flatten(actions, start_dim = 0, end_dim = 1)
        feedback = torch.flatten(feedback, start_dim = 0, end_dim = 1)
        all_actions = []
        all_feedback = []
        test_org = torch.load('test_org', map_location = torch.device('cpu'))
        test_org_first = [t[0] for t in test_org]

        def other_references(indices):
                i = test_org_first.index([int(x) for x in indices][:sentence_length(indices)+1])
                refs = test_org[i]
                to_return = []
                for ref in range(len(refs)):
                       to_return.append([i2w[x] for x in refs[ref]][1:])
                return(to_return)

        for i in range(len(actions)):
                all_actions.append([i2w[int(x)] for x in actions[i]])
                all_feedback.append(other_references(feedback[i]))
        all_actions = list(map(lambda x : x[1:sentence_length(x)], all_actions))
        return(bs.corpus_bleu(all_feedback, all_actions, smoothing_function=SmoothingFunction().method7))

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument("--pretrained-path", type=str, metavar='P', default=None)
        args = parser.parse_args()
        bleu = compute_bleu(args.pretrained_path)
        # print(args.pretrained_path)
        print(bleu)
