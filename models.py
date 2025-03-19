from transformers import PreTrainedModel, AutoConfig, BertModel, PretrainedConfig, AutoTokenizer
import torch
from huggingface_hub import login
from utils import preprocess
import warnings

warnings.filterwarnings("ignore")


class BinaryConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BinaryClassifier(PreTrainedModel):
    config_class = BinaryConfig

    def __init__(self, config):
        super().__init__(config)
        encoder_config = AutoConfig.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
        self.encoder = BertModel(encoder_config)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            embeddings = self.encoder(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )[1]
        else:
            embeddings = self.encoder(input_ids, attention_mask=attention_mask)[1]

        return self.classifier(embeddings)


def process_batch_binary_classification(texts, model, tokenizer, device):
    preprocessed_texts = [preprocess(text) for text in texts]
    model_inputs = tokenizer(
        preprocessed_texts, padding="max_length", truncation=True, max_length=120
    )
    model_inputs = {k: torch.tensor(v).to(device) for k, v in model_inputs.items()}
    with torch.no_grad():
        predictions = model(**model_inputs)
    predictions = predictions.detach().cpu().numpy()
    predictions = [int(x >= 0.0) for x in predictions]
    return predictions


def get_binary_model(model_path, tokenizer_path, hf_token=None):
    if hf_token:
        login(token=hf_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier.from_pretrained(model_path)
    model = model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer, device
