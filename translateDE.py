import os
import json
import pickle
import copy

# from googletrans import Translator, LANGUAGES
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map
from pathlib import Path
from attrdict import AttrDict

from data_loader import GoEmotionsProcessor

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/felix/masterthesis/local-storm-351416-b1a2b33b3abc.json"

import six
# from google.cloud import translate_v2 as translate


to_taxonomy = "ekman"
datapath = Path(f"/home/felix/masterthesis/datasets/GoEmiotionsData/data")

def translate_text(text, target="de"):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    return result["translatedText"]


def translate_dataset():

    # Read from config file and make args

    taxonomy = "original"

    config_filename = "{}.json".format(taxonomy)
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))

    print("Training/evaluation parameters {}".format(args))

    # Translation

    # text_a contains information, text_b always None
    for mode in tqdm(["dev", "test", "train"]): # ["dev", "test", "train"]

        processor = GoEmotionsProcessor(args)
        label_list_len = len(processor.get_labels())
        examples = processor.get_examples(mode)
        examples_de = copy.deepcopy(examples)
        examples_text_a = [str(e.text_a) for e in examples]

        ###### Translating using the google api test credits ######

        # examples_text_a_de = process_map(translate_text, examples_text_a, max_workers=os.cpu_count())
        #
        # for i in range(len(examples)):
        #     examples_de[i].text_a = examples_text_a_de[i]
        #
        # with open(f"/home/felix/PycharmProjects/GoEmotions-pytorch/data/{taxonomy}/{mode}DE.obj", "wb") as f:
        #     pickle.dump(examples_de, f)

        ###### Load translated obj #####

        # print("\n", mode, "\n\n")
        # with open(f"/home/felix/PycharmProjects/GoEmotions-pytorch/data/{taxonomy}/{mode}DE.obj", "rb") as f:
        #     examples_de = pickle.load(f)

        ##### Save in tsv format

        with open(datapath / to_taxonomy / (f"{mode}.tsv"), "r", encoding="utf-8") as f:
            examples_txt = [e.split("\t") for e in f.readlines()]

        with open(datapath / "original" / (f"{mode}DE.tsv"), "r", encoding="utf-8") as f:
            examples_txt_de = [e.split("\t") for e in f.readlines()]

        examples_txt_de_to_tax = []

        for i in range(len(examples_txt)):
            element = [examples_txt_de[i][0], examples_txt[i][1], examples_txt[i][2]]
            examples_txt_de_to_tax.append(element)

        for i, e_t in tqdm(enumerate(examples_txt_de_to_tax)):
            with open(datapath / to_taxonomy / (f"{mode}DE.tsv"), 'a') as the_file:
                the_file.write("\t".join(e_t))

        # for i, e_t in tqdm(enumerate(examples_txt)):
        #     e_t = e_t.split("\t")
        #     if examples_de[i].label == [int(n) for n in e_t[1].split(",")] and \
        #             i == int(examples_de[i].guid.split("-")[-1]):
        #         e_t[0] = examples_de[i].text_a
        #     else:
        #         print("debug")
        #     with open(f'/home/felix/PycharmProjects/GoEmotions-pytorch/data/original/{mode}DE.tsv', 'a') as the_file:
        #         the_file.write("\t".join(e_t))


        ###### Correct for weird tokens, manual check and correction if necessary ######

        # import re
        # pattern = r"\[[a-zA-ZäüöÄÜÖ]+\]"
        # print(set(re.findall(pattern, "".join([str(e.text_a) for e in examples]))))
        # print(set(re.findall(pattern, "".join([str(e.text_a) for e in examples_de]))))
        #
        # placeholders = set(re.findall(pattern, "".join([str(e.text_a) for e in examples_de])))
        #
        # for placeholder in placeholders:
        #     elements = {i:e for i, e  in enumerate(examples_de) if placeholder in e.text_a}
        #     print(placeholder, " has ", len(elements), "cases")
        #     for i in elements.keys():
        #         print(examples[i].text_a)
        #         print(examples_de[i].text_a)
        #         text = "Warum ist Indien nicht verboten? Eines der Länder mit den meisten [RELIGION] auf dem Planeten."
        #         examples_de[i].text_a = text
        #
        # with open(f"/home/felix/PycharmProjects/GoEmotions-pytorch/data/{taxonomy}/{mode}DE.obj", "wb") as f:
        #     pickle.dump(examples_de, f)

def load_bert_model():

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
    model = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")

    return tokenizer, model

if __name__ == "__main__":
    translate_dataset()
