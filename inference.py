import torch
from typing import Callable
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_device() -> torch.device:
    # return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device('cpu')

def preprocess_text(text, tokenizer, max_len=350, n_tokens=20):
    inputs = tokenizer.encode_plus(
        text,
        None,
        truncation=True,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_token_type_ids=True,
    )
    inputs['input_ids']=torch.cat((torch.full((1, n_tokens), 500).resize(n_tokens),torch.tensor(inputs['input_ids'], dtype=torch.long)))
    inputs['attention_mask'] = torch.cat((torch.full((1, n_tokens), 1).resize(n_tokens), torch.tensor(inputs['attention_mask'], dtype=torch.long)))
    return inputs

def setup_pipeline(model_path: str, model_name: str='bert-base-cased') -> Callable:

    device = get_device()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=350)
    eval_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    eval_model.load_state_dict(torch.load(model_path, map_location=device))
    eval_model.to(device)

    def pipeline(text):
        model_input = preprocess_text(text, tokenizer=tokenizer)
        with torch.no_grad():
            result = eval_model(
                model_input['input_ids'].unsqueeze(0).to(device), 
                model_input['attention_mask'].unsqueeze(0).to(device)
            )
        return {"score": result["logits"].item()}

    return pipeline



