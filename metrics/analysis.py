import argparse
from collections import Counter
import spacy
import matplotlib.pyplot as plt
import numpy as np

nlp = spacy.load("en_core_web_sm")

def calc_length(text: str) -> int:
    text_list = text.strip().split()
    i = 0
    for word in text_list:
        if word == '<EOS>':
            return i
        if word not in ['<PAD>', '<BOS>']:
            i += 1
    return i

def pos_count(model, text):
    length = calc_length(text)
    text = " ".join(text.strip().split()[:length])
    doc = model(text)
    pos = [token.pos_ for token in doc]
    return(Counter(pos))

def get_overlap(model, text, caption, POS):
    # gets number of <POS> in caption that are also present in text.
    length = calc_length(text)
    text = " ".join(text.strip().split()[:length])
    doc = model(caption)
    nouns = filter(lambda token : token.pos_ == POS, doc)
    nouns = [token.text in text for token in nouns]
    return(sum(nouns), len(nouns))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-file', type=str, default='output.log',
        help="Source File that we get data from")
    parser.add_argument('--max-length', type=int, default=99999999,
        help="Maximum number of steps to analyze")
    parser.add_argument('--bucket-size', type=int, default=10000,
        help="Aggregate by groups of N sentences")
    parser.add_argument('--output-file', type=str, default="output.png",
        help="Base path for output file")
    parser.add_argument('--caption-measure', type=str, default="recall",
        help="'recall', 'precision', or 'F1'")
    parser.add_argument("--title", type=str, default="",
        help="Title for Plots")
    parser.add_argument("--spacing", type=int, default=125,
        help="Spacing Between Returns")
    parser.add_argument("--ignore-repeats", type=lambda x:bool(strtobool(x)), default=False, 
        nargs='?', const=True, help="Whether to ignore repeated words")

    args = parser.parse_args()
    bucket = args.bucket_size
    with open(args.source_file, "r") as f:

        POS_list = ["ADJ", "NOUN", "PROPN", "VERB"]
        lines = f.readlines()
        length_data = []
        overlap_data = [[] for pos in POS_list]
        caption_pos_data = [[] for pos in POS_list]
        pos_data = [[] for pos in POS_list]
        count = 0

        for idx, line in enumerate(lines[:-2]):
            if count > args.max_length:
                break
              
            if lines[idx+2].startswith("SPS"):
                # check for repeats
                caption = lines[idx+1]
                if args.ignore_repeats:
                    line = ' '.join(set(line.split()))
                    caption = ' '.join(set(caption.split()))
                count += 1
                # get length  
                length_data.append(calc_length(line))
                # get POS data
                pos = pos_count(nlp, line)
                for i in range(len(POS_list)):
                    pos_data[i].append(pos[POS_list[i]])
                # get overlap data
                for i in range(len(POS_list)):
                    overlap = get_overlap(nlp, line, caption, POS_list[i])
                    overlap_data[i].append(overlap[0])
                    caption_pos_data[i].append(overlap[1])

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19.2,4.8))
        fig.suptitle(args.title)

        # plot length
        new_length_data = [sum(length_data[idx-bucket:idx])/bucket for idx in range(bucket, len(length_data))] 
        ax1.plot(list(range(args.spacing*bucket, args.spacing*len(length_data), args.spacing)), new_length_data)
        ax1.title.set_text("Utterance Length over Time")

        # plot POS data
        to_plot = [[] for i in range(len(POS_list))]
        for i in range(len(POS_list)):
            pos = [sum(pos_data[i][idx-bucket:idx])/sum(length_data[idx-bucket:idx]) for idx in range(bucket, len(overlap_data[0]))]
            to_plot[i] = pos
        ax2.plot(list(range(args.spacing*bucket, args.spacing*len(overlap_data[0]), args.spacing)), np.transpose(np.array(to_plot)), label=POS_list)
        ax2.title.set_text("POS Frequency over Time")
        ax2.legend()

        # plot POS recall/frequency/f1 data
        pos_overlap_data = []
        if args.caption_measure == "recall":
            for i in range(len(POS_list)):
                overlap = [sum(overlap_data[i][idx-bucket:idx])/sum(caption_pos_data[i][idx-bucket:idx]) for idx in range(bucket, len(overlap_data[0]))]
                pos_overlap_data.append(overlap)
        elif args.caption_measure == "accuracy":
            for i in range(len(POS_list)):
                overlap = [sum(pos_data[i][idx-bucket:idx])/sum(caption_pos_data[i][idx-bucket:idx]) for idx in range(bucket, len(overlap_data[0]))]
                pos_overlap_data.append(overlap)
        else: #F1
            for i in range(len(POS_list)):
                recall = [sum(overlap_data[i][idx-bucket:idx])/sum(caption_pos_data[i][idx-bucket:idx]) for idx in range(bucket, len(overlap_data[0]))]
                precision = [sum(pos_data[i][idx-bucket:idx])/sum(caption_pos_data[i][idx-bucket:idx]) for idx in range(bucket, len(overlap_data[0]))]
                overlap = [2*recall[idx]*precision[idx]/(recall[idx] + precision[idx]) for idx in range(min(len(recall), len(precision)))]
                pos_overlap_data.append(overlap)           

        ax3.plot(list(range(args.spacing*bucket, args.spacing*len(overlap_data[0]), args.spacing)), np.transpose(np.array(pos_overlap_data)), label=POS_list)
        ax3.title.set_text(f"Caption {args.caption_measure} by POS over Time")
        ax3.legend()
        
        plt.savefig(args.output_file)