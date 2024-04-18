import os
import pandas as pd
import numpy as np

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


SEED = 42
DATAPATH = 'data' # set path to your data
VAL_SIZE = 0.2
TARGET = 'target'
MODEL_NAME = 'bert-base-uncased'
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def tokenize(batch):
    return tokenizer(batch['excerpt'], padding='max_length', truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    metric = rmse(labels, predictions)
    return {"rmse": metric}


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(DATAPATH,'train.csv'))
    data = data[['excerpt', TARGET]]
    data = data.rename(columns={TARGET: "labels"})

    np.random.seed(SEED)
    train_df, val_df = train_test_split(data, test_size=VAL_SIZE, random_state=SEED)
    scaler = StandardScaler()
    train_df['labels'] = scaler.fit_transform(train_df[['labels']])

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False) 
    train_dataset = train_dataset.train_test_split(test_size=0.2)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False) 

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=350)

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["excerpt"])
    val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["excerpt"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1
    )

    training_args = TrainingArguments(
        output_dir="trainer",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-5,
        num_train_epochs=10,
        save_strategy='epoch',
        seed=SEED,
        report_to='none',
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    val_preds = trainer.predict(val_dataset)
    val_preds= scaler.inverse_transform(pd.DataFrame(val_preds[0].reshape(1,-1)[0]))

    print(f'RMSE: {rmse(val_df["labels"].values, val_preds.reshape(1,-1)[0])}')