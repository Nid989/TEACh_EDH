import torch
import numpy as np
import datasets
from datasets import Dataset, load_dataset, load_from_disk
import pandas as pd
import nltk

from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    LlamaForCausalLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

nltk.download("punkt", quiet=True)
metric = datasets.load_metric("rouge")

def main():
    
    model_name = "meta-llama/Llama-2-13b-hf"
    model = LlamaForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preprocess the dataset 
#     train_data, validation_data = preprocess_data(tokenizer)

    train_data = load_from_disk('../train_data_arrow')
#     validation_data = load_from_disk('../valid_data_arrow')
    
    tuner = tune_model(train_data, tokenizer, model)
    tuner.train()
    
    
    
    pass


################ Preprocessing functions ####################

def preprocess_data(tokenizer):

    # Load the dataset with only Dialog History
    train_df = pd.read_csv('llms/gameplan_data/train_dh.csv')
    valid_seen_df = pd.read_csv('llms/gameplan_data/valid_seen_dh.csv')
    
    # Convert into 'Dataset' format
    train_data_txt = Dataset.from_pandas(train_df)
    validation_data_txt = Dataset.from_pandas(valid_seen_df)
    
    train_data = train_data_txt.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, 1024, 1024
        ),
        batched=True,
        remove_columns=train_data_txt.column_names,
    )

    validation_data = validation_data_txt.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, 1024, 1024
        ),
        batched=True,
        remove_columns=validation_data_txt.column_names,
    )
    
    return train_data, validation_data
    
def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    tokenizer.pad_token = tokenizer.eos_token
    source, target = batch["dialog"], batch["gameplan_prediction"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    
    return batch  

##########################Evaluation functions#####################

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


###########################Fine-Tuning function############################

def tune_model(train_data, tokenizer, model):
    tokenizer.pad_token = tokenizer.eos_token
    training_args = Seq2SeqTrainingArguments(
        output_dir="../Llama2-13b-DH",
        num_train_epochs=1, 
        do_train=True,
        per_device_train_batch_size=4, 
        learning_rate=3e-05,
        warmup_steps=500,
        weight_decay=0.1,
        label_smoothing_factor=0.1,
        predict_with_generate=True,
        logging_dir="logs",
        logging_steps=50,
        save_total_limit=3,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    return trainer

if __name__=='__main__':
    main()