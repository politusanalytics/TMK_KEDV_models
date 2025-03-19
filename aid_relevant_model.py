import torch
from models import BinaryClassifier, process_batch_binary_classification, get_model
from utils import preprocess


if __name__ == "__main__":
    texts = ["Bu bir test cümlesidir", "Bu da bir test cümlesidir"]
    model_path = "Politus/oxfam_help-binary-dbmdz-bert-base-turkish-128k-uncased-128"
    tokenizer_path = "dbmdz/bert-base-turkish-128k-uncased"
    model, tokenizer, device = get_model(model_path, tokenizer_path)
    predictions = process_batch_binary_classification(texts, model, tokenizer, device)
    print(predictions)
