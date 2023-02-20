from builtins import set
from datasets import load_dataset
from collections import Counter
import sys


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(dataset=load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT"))
def find_image(sentence):
    urls = find_image.dataset["train"]["URL"]
    texts = find_image.dataset["train"]["TEXT"]

    results = []
    for idx, i in enumerate(texts):
        if sentence in i.lower():
            results.append(urls[idx])
    return results

def find_image_multi(sentences):
    urls = find_image.dataset["train"]["URL"]
    texts = find_image.dataset["train"]["TEXT"]
    sentence = sentences[0]
    results = Counter(find_image(sentence))
    for sentence_index in range(1,len(sentences)):
        results += Counter(find_image(sentences[sentence_index]))
    return(results.most_common(1)[0][0])

def build_caption_to_image_dict(dataset_folder):
    import json
    captions = json.load(open(f"{dataset_folder}/annotations/captions_train2017.json"))
    id2filename = dict()
    for image in captions["images"]:
        id2filename[image["id"]] = f'{dataset_folder}/train2017/{image["file_name"]}'
    ret = {i['caption'].lower(): id2filename[i['image_id']] for i in captions["annotations"]}   
    captions = json.load(open(f"{dataset_folder}/annotations/captions_val2017.json"))
    id2filename = dict()
    for image in captions["images"]:
        id2filename[image["id"]] = f'{dataset_folder}/val2017/{image["file_name"]}'
    ret.update({i['caption'].lower(): id2filename[i['image_id']] for i in captions["annotations"]})
    return ret


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentences = [sys.argv[1]]
    else:
        sentences = ['a very clean white bathroom with blue', 'a bathroom vanity mirror above a bathroom', 'bathroom sink with mirror , lighting ,', 'a bathroom sink area is lit up', 'a bathroom with a white counter top']

    print(find_image_multi(sentences))