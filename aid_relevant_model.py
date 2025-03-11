from transformers import PreTrainedModel, AutoConfig, AutoTokenizer, BertModel
import torch
from huggingface_hub import login
from transformers import PretrainedConfig
import json
import torch.nn as nn
import sys
import time
from pprint import pprint
from transformers import (
    PreTrainedModel,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertModel,
)
import torch
from huggingface_hub import login
from transformers import PretrainedConfig

login(token="Your token goes here")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Currently running on {device}")


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


sector_names = ["oxfam_help"]


class BinarySectorConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BinarySectorClassifier(PreTrainedModel):
    config_class = BinarySectorConfig

    def __init__(self, config):
        super().__init__(config)
        encoder_config = AutoConfig.from_pretrained(
            "dbmdz/bert-base-turkish-128k-uncased"
        )
        self.encoder = BertModel(encoder_config)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, input_mask, token_type_ids=None):
        if token_type_ids is not None:
            with torch.no_grad():
                embeddings = self.encoder(
                    input_ids, attention_mask=input_mask, token_type_ids=token_type_ids
                )[1]
        else:
            with torch.no_grad():
                embeddings = self.encoder(input_ids, attention_mask=input_mask)[1]

        return self.classifier(embeddings)


BATCH_SIZE = 1536 * 8
model = BinarySectorClassifier.from_pretrained(
    "Politus/oxfam_help-binary-dbmdz-bert-base-turkish-128k-uncased-128"
)


model.eval()
model = model.cuda()

model = torch.nn.DataParallel(model, device_ids=[i for i in range(8)])

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")

import pymongo
from pymongo import UpdateOne

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["politus_twitter"]
tweet_col = db["tweets"]

# tweets_to_predict = tweet_col.find(projection = ["_id", "text"])
# tweets_to_predict = tweet_col.find({"sector_labels": None}, ["_id", "text"])


# with open("/data01/myardi/database_exports/240606_org/org_tweet_ids_240606.txt", "r") as file:
#     line = file.read().strip()
#     all_ids = line.split(",")
#     print(len(all_ids))

data = []

with open(
    "./all_relevant_user_tweets_info_deprem_predictions_06_28_2024.jsonl", "r"
) as file:
    for line in file:
        tweet_dict = json.loads(line)
        if tweet_dict.get("deprem_prediction") == 1:
            data.append(tweet_dict)
print(len(data))

total_processed = 0
curr_batch_ids = []
curr_batch_texts = []
curr_deprem_predictions = []
curr_batch_count = 0

output_file = open(
    "all_relevant_user_tweets_info_binary_aid_predictions_06_28_2024.jsonl",
    "w",
    encoding="utf-8",
)

for i, tweet in enumerate(data):
    id_str = tweet.get("id")
    text = tweet.get("text")
    deprem_pred = tweet.get("deprem_prediction")

    if len(text) > 0:
        curr_batch_ids.append(id_str)
        curr_batch_texts.append(text)
        curr_deprem_predictions.append(deprem_pred)
        curr_batch_count += 1

    if curr_batch_count == BATCH_SIZE:
        model_inputs = tokenizer(
            curr_batch_texts, padding="max_length", truncation=True, max_length=120
        )
        input_ids = model_inputs["input_ids"]
        token_type_ids = model_inputs["token_type_ids"]
        attention_mask = model_inputs["attention_mask"]

        output = model(
            torch.tensor(input_ids),
            torch.tensor(attention_mask),
            torch.tensor(token_type_ids),
        )
        preds = output.detach().cpu().numpy()
        preds = [int(x >= 0.0) for x in preds]

        for i in range(curr_batch_count):
            predicted_tweet = dict()
            predicted_tweet["id"] = curr_batch_ids[i]
            predicted_tweet["text"] = curr_batch_texts[i]
            predicted_tweet["deprem_prediction"] = curr_deprem_predictions[i]
            predicted_tweet["help_prediction"] = preds[i]
            output_file.write(json.dumps(predicted_tweet) + "\n")

        curr_batch_ids = []
        curr_batch_texts = []
        curr_batch_count = 0
        total_processed += BATCH_SIZE
        print(f"Total processed tweets: {total_processed}")

if curr_batch_count > 0:
    model_inputs = tokenizer(
        curr_batch_texts, padding="max_length", truncation=True, max_length=120
    )
    input_ids = model_inputs["input_ids"]
    token_type_ids = model_inputs["token_type_ids"]
    attention_mask = model_inputs["attention_mask"]

    output = model(
        torch.tensor(input_ids),
        torch.tensor(attention_mask),
        torch.tensor(token_type_ids),
    )
    preds = output.detach().cpu().numpy()
    preds = [int(x >= 0.0) for x in preds]

    for i in range(curr_batch_count):
        predicted_tweet = dict()
        predicted_tweet["id"] = curr_batch_ids[i]
        predicted_tweet["text"] = curr_batch_texts[i]
        predicted_tweet["deprem_prediction"] = curr_deprem_predictions[i]
        predicted_tweet["help_prediction"] = preds[i]
        output_file.write(json.dumps(predicted_tweet) + "\n")

    curr_batch_ids = []
    curr_batch_texts = []
    total_processed += curr_batch_count
    print(f"Total processed tweets: {total_processed}")
