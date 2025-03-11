from transformers import PreTrainedModel, AutoConfig, AutoTokenizer, BertModel
import torch
from huggingface_hub import login
from transformers import PretrainedConfig
import json
import torch.nn as nn
import sys
import time

login(token="YOUR TOKEN GOES HERE")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_list = [
    "Basic \nNeeds",
    "Nutrition",
    "Water \nSanitation\nHygiene",
    "Shelter",
    "Livelihood\nSupport",
    "Healthcare",
    "Search \nand \nRescue",
    "Logistics & \nEmergency\nInfrastructure",
    "Education",
    "Psycho-social-Support-and-protection",
]


class MultiLabelSectorConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        super().__init__(**kwargs)


config_file = MultiLabelSectorConfig(10)


class MultiLabelSectorClassifier(PreTrainedModel):
    config_class = MultiLabelSectorConfig

    def __init__(self, config):
        super().__init__(config)
        encoder_config = AutoConfig.from_pretrained(
            "dbmdz/bert-base-turkish-128k-uncased"
        )
        self.encoder = BertModel(encoder_config)
        self.classifier = torch.nn.Linear(
            self.encoder.config.hidden_size, config.num_classes
        )
        self.label_list = [
            "Basic \nNeeds",
            "Nutrition",
            "Water \nSanitation\nHygiene",
            "Shelter",
            "Livelihood\nSupport",
            "Healthcare",
            "Search \nand \nRescue",
            "Logistics & \nEmergency\nInfrastructure",
            "Education",
            "Psycho-social-Support-and-protection",
        ]

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
    ):
        if token_type_ids is not None:
            with torch.no_grad():
                embeddings = self.encoder(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )[1]
        else:
            with torch.no_grad():
                embeddings = self.encoder(input_ids, attention_mask=attention_mask)[1]

        return self.classifier(embeddings)


# model = MultiLabelSectorClassifier(config_file)


BATCH_SIZE = 1536 * 8
model = MultiLabelSectorClassifier.from_pretrained(
    "Politus/multi-label-oxfam-help-classifier-dbmdz-bert-base-turkish-128k-uncased-120"
)
model.eval()
model = model.cuda()

model = torch.nn.DataParallel(model, device_ids=[i for i in range(8)])

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")

data = []

with open(
    "./all_relevant_user_tweets_info_binary_aid_predictions_06_28_2024.jsonl", "r"
) as file:
    for line in file:
        tweet_dict: dict = json.loads(line)
        if tweet_dict.get("help_prediction") == 1:
            data.append(tweet_dict)

print(len(data))

total_processed = 0
curr_batch_ids = []
curr_batch_texts = []
curr_batch_count = 0

output_file = open(
    "all_relevant_user_tweets_info_aid_classification_predictions_06_28_2024.jsonl",
    "w",
    encoding="utf-8",
)


for i, tweet in enumerate(data):
    id_str = tweet.get("id")
    text = tweet.get("text")

    if len(text) > 0:
        curr_batch_ids.append(id_str)
        curr_batch_texts.append(text)
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
        preds = [[int(y >= 0.0) for y in x] for x in preds]

        for ii in range(curr_batch_count):
            predicted_tweet = dict()
            predicted_tweet["id"] = curr_batch_ids[ii]
            predicted_tweet["text"] = curr_batch_texts[ii]
            predicted_tweet["help_prediction"] = 1

            other = 1
            for j, label in enumerate(label_list):
                predicted_tweet[label] = preds[ii][j]
                if preds[ii][j] == 1:
                    other = 0
            predicted_tweet["Other"] = other

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
    preds = [[int(y >= 0.0) for y in x] for x in preds]

    for ii in range(curr_batch_count):
        predicted_tweet = dict()
        predicted_tweet["id"] = curr_batch_ids[ii]
        predicted_tweet["text"] = curr_batch_texts[ii]
        predicted_tweet["help_prediction"] = 1

        other = 1
        for j, label in enumerate(label_list):
            predicted_tweet[label] = preds[ii][j]
            if preds[ii][j] == 1:
                other = 0
        predicted_tweet["Other"] = other

        output_file.write(json.dumps(predicted_tweet) + "\n")

    curr_batch_ids = []
    curr_batch_texts = []
    total_processed += curr_batch_count
    curr_batch_count = 0
    print(f"Total processed tweets: {total_processed}")
