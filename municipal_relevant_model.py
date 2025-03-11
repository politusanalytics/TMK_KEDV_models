from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer
import torch
from huggingface_hub import login

login(token="")
device = torch.device("cuda")


class BinaryBertClassification(PreTrainedModel):
    def __init__(self, config_file = "dbmdz/bert-base-turkish-128k-uncased"):
        config = AutoConfig.from_pretrained(config_file)
        super().__init__(config)

        self.encoder = AutoModel.from_pretrained(config_file)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, 1)

        self.idx_to_label = {0: "Irrelevant", 1: "Relevant"}
    
    def forward(self, input_ids, attention_mask, token_type_ids = None):
        if token_type_ids is not None:
            with torch.no_grad():
                embeddings = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[1]
        else:
            with torch.no_grad():
                embeddings = self.encoder(input_ids, attention_mask=attention_mask)[1]
        
        out = self.classifier(embeddings)
        preds = torch.sigmoid(out).detach().cpu().numpy().flatten()
        preds = [self.idx_to_label[int(x >= 0.5)] for x in preds]

        return preds


model = BinaryBertClassification("Politus/MunicipalRelevant-dbmdz-bert-base-turkish-128k-uncased").to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
config = AutoConfig.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
has_token_type_ids = config.type_vocab_size > 1

input = tokenizer("Ben bu belediye işlerini sevdim.")
print(input)
input_ids = input['input_ids']
token_type_ids = input['token_type_ids']
attention_mask = input['attention_mask']

output = model(input_ids = torch.unsqueeze(torch.tensor(input_ids).to(device), dim = 0), attention_mask = torch.unsqueeze(torch.tensor(attention_mask).to(device), dim = 0), token_type_ids = torch.unsqueeze(torch.tensor(token_type_ids).to(device), dim = 0))

print(output)
