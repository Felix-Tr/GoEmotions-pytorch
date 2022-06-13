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
    import pandas as pd
    # stats_df = pd.DataFrame(columns=["mode", "anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust"])
    # Translation/Conversion

    # text_a contains information, text_b always None
    for mode in tqdm(["dev", "test", "train"]): # ["dev", "test", "train"]

        # processor = GoEmotionsProcessor(args)
        # label_list_len = len(processor.get_labels())
        # examples = processor.get_examples(mode)
        # examples_de = copy.deepcopy(examples)
        # examples_text_a = [str(e.text_a) for e in examples]

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

        # with open(datapath / to_taxonomy / (f"{mode}.tsv"), "r", encoding="utf-8") as f:
        #     examples_txt_ekman = [e.split("\t") for e in f.readlines()]
        #
        # with open(datapath / "original" / (f"{mode}DE.tsv"), "r", encoding="utf-8") as f:
        #     examples_txt_de = [e.split("\t") for e in f.readlines()]

        examples_txt_de_to_tax = []

        # for i in range(len(examples_txt_ekman)):
        #     element = [examples_txt_de[i][0], examples_txt_ekman[i][1], examples_txt_ekman[i][2]]
        #     examples_txt_de_to_tax.append(element)
        #
        # for i, e_t in tqdm(enumerate(examples_txt_de_to_tax)):
        #     with open(datapath / to_taxonomy / (f"{mode}DE.tsv"), 'a') as the_file:
        #         the_file.write("\t".join(e_t))

        ################# aNALYSE LABEL STATISTICS FOR EKMANN AND ORIGINAL

        # to_taxonomy = "ekman"
        # for to_taxonomy in ["ekman", "original"]:
        #     datapath = Path(f"/content/drive/MyDrive/temporary/masterthesis_drive/GoEmiotionsData/data")
        #
        #     if to_taxonomy == "original":
        #         label_order = ["anger", "excitement", "disgust", "fear", "joy", "love", "optimism", "disapproval","disappointment", "sadness", "surprise", "approval","caring"]
        #     else:
        #         label_order = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        #     with open(datapath / to_taxonomy / (f"{mode}.tsv"), "r", encoding="utf-8") as f:
        #         examples_txt_ekman = [e.split("\t") for e in f.readlines()]
        #     with open(datapath / to_taxonomy / (f"labels.txt_ek")) as f:
        #         labels = {l.replace("\n", ""):i for i, l in enumerate(f.readlines())}
        #
        #     row = {"mode": to_taxonomy+"-"+mode}
        #     for i, label in enumerate(label_order):
        #         row[label] = sum([labels[label] in [int(t_) for t_ in t[1].split(",")] for t in examples_txt_ekman])
        #     stats_df = stats_df.append(row, ignore_index=True)


        ################# Convert to SpanEmo Format v2 including missing labels

        to_taxonomy = "ekman"


        label_order = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust"]
        for lang in ["", "DE"]:
            datapath = Path(f"/content/drive/MyDrive/temporary/masterthesis_drive/GoEmiotionsData/data")
            with open(datapath / "ekman" / (f"{mode}{lang}.tsv"), "r", encoding="utf-8") as f:
                examples_txt_ekman = [e.split("\t") for e in f.readlines()]
            with open(datapath / "original" / (f"{mode}{lang}.tsv"), "r", encoding="utf-8") as f:
                examples_txt_original = [e.split("\t") for e in f.readlines()]

            with open(datapath / "ekman" / (f"labels.txt")) as f:
                labels_ek = {i:l.replace("\n", "") for i, l in enumerate(f.readlines())}
            with open(datapath / "original" / (f"labels.txt")) as f:
                labels_o = {i:l.replace("\n", "") for i, l in enumerate(f.readlines())}

            rows = []
            for txt_ek, txt_o in zip(examples_txt_ekman, examples_txt_original):
                text_ek, labels_temp_ek, id_ek = txt_ek[0], [int(n) for n in txt_ek[1].split(",")], txt_ek[2].replace("\n", "")
                text_o, labels_temp_o, id_o = txt_ek[0], [int(n) for n in txt_ek[1].split(",")], txt_ek[2].replace("\n", "")
                row_labels = []
                labels_temp_ek = [labels_ek[i] for i in labels_temp_ek]
                labels_temp_o = [labels_ek[i] for i in labels_temp_o]
                row_labels = []
                for label in label_order:
                    if label == "anticipation":
                        present = "excitement" in labels_temp_o
                    elif label == "pessimism":
                        present = ("disapproval" in labels_temp_o) or ("disappointment" in labels_temp_o)
                    elif label == "trust":
                        present = ("approval" in labels_temp_o) or ("caring" in labels_temp_o)
                    else:
                        present = label in labels_temp_ek
                    row_labels.append(str(int(present)))
                row = [id_o, text_o] + row_labels
                rows.append(row)

            datapath = Path(f"/content/drive/MyDrive/temporary/masterthesis_drive/SpanEmoData/E-c")
            mode = "test-gold" if mode == "test" else mode
            lang = "-DE" if lang == "DE" else lang
            with open(datapath / (f"GoEmotions-{mode}{lang}.txt"), 'a') as file:
                file.write("ID\tTweet\tanger\tanticipation\tdisgust\tfear\tjoy\tlove\toptimism\tpessimism\tsadness\tsurprise\ttrust\n")
            for row in rows:
                with open(datapath / (f"GoEmotions-{mode}{lang}.txt"), 'a') as file:
                    file.write("\t".join(row) + "\n")
            mode = "test" if mode == "test-gold" else mode
            lang = "DE" if lang == "-DE" else lang

        # for i, e_t in tqdm(enumerate(examples_txt_ekman)):
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

    # stats_df.to_csv(datapath / (f"label-stats.csv"), index=False)

def load_bert_model():

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
    model = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")

    return tokenizer, model

if __name__ == "__main__":
    translate_dataset()
