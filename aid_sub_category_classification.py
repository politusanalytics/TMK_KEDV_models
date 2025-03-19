from transformers import PreTrainedModel, AutoConfig, AutoTokenizer, BertModel
import torch
from huggingface_hub import login
from transformers import PretrainedConfig
from my_token import token
from utils import preprocess
import warnings

warnings.filterwarnings("ignore")


class MultiLabelConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(self, num_classes=10, **kwargs):
        self.num_classes = num_classes
        super().__init__(**kwargs)


class MultiLabelClassifier(PreTrainedModel):
    config_class = MultiLabelConfig

    def __init__(self, config):
        super().__init__(config)
        encoder_config = AutoConfig.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
        self.encoder = BertModel(encoder_config)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        if token_type_ids is not None:
            embeddings = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )[1]
        else:
            embeddings = self.encoder(input_ids, attention_mask=attention_mask)[1]

        return self.classifier(embeddings)


def process_batch_multi_label_classification(texts, model, tokenizer, device):
    model_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=120)
    input_ids = model_inputs["input_ids"].to(device)
    token_type_ids = model_inputs["token_type_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)

    output = model(
        torch.tensor(input_ids),
        torch.tensor(attention_mask),
        torch.tensor(token_type_ids),
    )
    preds = output.detach().cpu().numpy()
    preds = [[int(y >= 0.0) for y in x] for x in preds]

    for i in range(len(texts)):
        predictions = dict()
        other = 1
        for j, label in enumerate(label_list):
            predictions[label] = preds[i][j]
            if preds[i][j] == 1:
                other = 0
        predictions["Other"] = other
    return preds


if __name__ == "__main__":

    login(token=token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_list = [
        "Basic Needs",
        "Nutrition",
        "Water Sanitation Hygiene",
        "Shelter",
        "Livelihood Support",
        "Healthcare",
        "Search and Rescue",
        "Logistics & Emergency Infrastructure",
        "Education",
        "Psycho-social Support and Protection",
    ]
    config_file = MultiLabelConfig(len(label_list))

    model = MultiLabelClassifier.from_pretrained(
        "Politus/multi-label-oxfam-help-classifier-dbmdz-bert-base-turkish-128k-uncased-120"
    )
    model.eval()
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")

    texts = [
        "Yardıma ihtiyacımız var, çok fazla yaralı var.",
        "Lütfen yardım edin, su ihtiyacımız var.",
        "Deprem sonrası bir çok insan evsiz kaldı.",
        "Deprem sonrası bir çok insan işsiz kaldı.",
        "Yardım edin piskolojim çok bozuk.",
    ]

    texts = [preprocess(text) for text in texts]

    predictions = process_batch_multi_label_classification(texts, model, tokenizer, device)
    print(predictions)
