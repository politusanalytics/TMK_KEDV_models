from transformers import (
    PreTrainedModel,
    AutoConfig,
    AutoTokenizer,
    BertModel,
    PretrainedConfig
)
import torch
from huggingface_hub import login

model_path = "Politus/oxfam_help-binary-dbmdz-bert-base-turkish-128k-uncased-128"
base_model = "dbmdz/bert-base-turkish-128k-uncased"

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


class BinaryConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BinaryClassifier(PreTrainedModel):
    config_class = BinaryConfig

    def __init__(self, config):
        super().__init__(config)
        encoder_config = AutoConfig.from_pretrained(base_model)
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


model = BinaryClassifier.from_pretrained(model_path)
model = model.eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(base_model)

def process_batch(texts):
    model_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=120)
    model_inputs = {k: torch.tensor(v).to(device) for k, v in model_inputs.items()}
    predictions = model(**model_inputs)
    predictions = predictions.detach().cpu().numpy()
    predictions = [int(x >= 0.0) for x in predictions]
    return predictions

if __name__ == "__main__":
    texts = ["Bu bir test cümlesidir", "Bu da bir test cümlesidir"]
    print(process_batch(texts))