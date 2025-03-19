import torch
from models import BinaryClassifier, process_batch_binary_classification, get_model
from utils import preprocess
from my_token import token

if __name__ == "__main__":
    texts = ["7.1 büyüklüğündeki felakat bir çok can aldı.", "Bu da bir test cümlesidir"]
    model_path = "Politus/earthquake-binary-dbmdz-bert-base-turkish-128k-uncased-128"
    tokenizer_path = "dbmdz/bert-base-turkish-128k-uncased"
    model, tokenizer, device = get_model(model_path, tokenizer_path, token)
    predictions = process_batch_binary_classification(texts, model, tokenizer, device)
    print(predictions)
