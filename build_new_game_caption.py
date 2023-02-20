import torch
from torch import nn
import argparse
import numpy as np
from collections import Counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-file', type=str, default='game_file.pt',
        help="Game File that we build from")
    parser.add_argument('--caption-weight', metavar='w', type=float, default=0.0, 
        help="weight of caption (in [0.0,1.0])")
    parser.add_argument('--similarity-model', type=str, default=None,
        help="what sentence transformer model to use to compute caption similarity, or 'none' for just word frequency")
    parser.add_argument('--output-path', type=str, default='new_game_file.pt',
        help="output path to save new game file")
    parser.add_argument('--tfidf', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help="Toggles whether to weight by TF-IDF when computing sentence embeddings")
    args = parser.parse_args()

    a = torch.load(args.source_file)
    images = torch.Tensor(a["images"])
    cos_image = torch.mm(images, images.t()) \
        / torch.mm(
            torch.linalg.vector_norm(images, dim=1).unsqueeze(1),
            torch.linalg.vector_norm(images, dim=1).unsqueeze(0)
        )
    if args.caption_weight == 0.0: # only use image similarity (default behavior)
        cos = cos_image
        cos_rank = torch.argsort(cos, descending=True)
        a["similarity_rank"] = cos_rank
        torch.save(a, args.output_path)

    else:
        import torch
        i2w = torch.load('i2w')
        if args.tfidf:
            captions = torch.Tensor(a["captions"])
            # get document frequency
            df_counter = Counter()
            for caption in captions:
                for item in caption:
                    df_counter[int(item)] += 1
            df_counter[0] = 999999 # extra large bc we don't want <PAD> --> size to influence anything
            # compute embeddings weighted by idf (equivalent to weighting by tf-idf)
            # caption = captions[0]
            # print(np.array([model.encode(i2w[int(i)]) for i in caption]).shape, np.transpose(np.array([idf_counter[int(i)]/total_count for i in caption])).shape)
            # new_list_of_lists = [np.matmul(np.transpose(np.array([model.encode(i2w[int(i)]) for i in caption])), np.transpose(np.array([idf_counter[int(i)]/total_count for i in caption]))) for caption in captions]
            if args.similarity_model is not None:
                # use sentence embeddings from similarity model
                i2w = torch.load("i2w")
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(args.similarity_model)
                new_list_of_lists = [np.matmul(np.transpose(np.array(model.encode([i2w[int(i)] for i in caption]))), np.transpose(np.array([1/df_counter[int(i)] for i in caption]))) for caption in captions]
                captions = torch.from_numpy(np.array(new_list_of_lists))
            else:
                tf_counters = []
                for caption in captions:
                    caption_counter = Counter()
                    for item in caption:
                        caption_counter[int(item)] += 1
                    caption_counter[0] = 0 # similarly ignoring effect of <PAD>/size
                    tf_counters.append(caption_counter)
                new_list_of_lists = [[tf_counters[idx][i]/df_counter[i] for i in df_counter.keys()] for idx, caption in enumerate(captions)]
                captions = torch.from_numpy(np.array(new_list_of_lists))

        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(args.similarity_model)
            captions = torch.Tensor(a["captions"])
            new_list_of_lists = [model.encode(" ".join([i2w[int(i)] for i in caption])) for caption in captions]
            captions = torch.from_numpy(np.array(new_list_of_lists))

        cos_caption = torch.mm(captions, captions.t()) \
            / torch.mm(
                torch.linalg.vector_norm(captions, dim=1).unsqueeze(1),
                torch.linalg.vector_norm(captions, dim=1).unsqueeze(0)
            )
        cw = args.caption_weight
        cos = torch.mul(cos_caption, cw) + torch.mul(cos_image, 1-cw)
        cos_rank = torch.argsort(cos, descending=True)
        a["similarity_rank"] = cos_rank
        torch.save(a, args.output_path)
