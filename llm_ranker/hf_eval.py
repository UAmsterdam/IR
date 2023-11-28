import torch
from transformers import BertForSequenceClassification, BertTokenizer, EarlyStoppingCallback
from datasets import load_dataset
import evaluate
from transformers import Trainer, TrainingArguments
import numpy as np
import torch.nn as nn
from datasets import Dataset
import subprocess
from collections import defaultdict
import sys

model_name = sys.argv[1]
# Define the model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)  # 1 label for binary classification

test_dataset_name = 'data/msmarco/dl2020_54.bm25.passage.top-1k.hf'
test_dataset_name = 'data/msmarco/dl2020_54.bm25.passage.ref.hf'
# Configure data files
test_dataset = Dataset.load_from_disk(test_dataset_name)
# entire dev set 
#eval_dataset = Dataset.load_from_disk('data/msmarco/msmarco.dev.bm25.passage.top-1k.hf')
# samples dev set
eval_dataset = Dataset.load_from_disk('data/msmarco/msmarco.dev.bm25.passage.top-1k.100.hf')

qrel_file_test = 'data/msmarco/2020qrels-pass.txt'
qrel_file_eval = 'data/msmarco/qrels.dev.tsv'
test_dataset.qrel = qrel_file_test 
eval_dataset.qrel = qrel_file_eval


# Define a function to preprocess and tokenize the data
def collate_fn(examples):
    question = [e["query"] for e in examples]
    doc = [e["document"] for e in examples]
    labels = torch.ones(len(examples))
    input_tokenized = tokenizer(question, doc, padding=True, truncation='only_second', max_length=256, return_tensors='pt')

    return {
            "inputs": input_tokenized,
            "labels": labels, 
            }


# Define a custom Trainer class
class CustomTrainer(Trainer):
    # Define the loss function for binary classification (cross-entropy)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ce = nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        logits = model(**inputs['inputs']).logits[:, 0]
        logits_plus_one = torch.concat((torch.Tensor([0]).to(logits.device), logits), 0)
        return torch.Tensor([[1]]), logits_plus_one 

def eval_trec(predictions, qids, dids, qrel_file, save_metrics=['map']):
    tmp_ranking_file = 'ranking.trec'
    with open(tmp_ranking_file, 'w') as f_out:
        for qid, did, score in zip(qids, dids, predictions):
            f_out.write(f'{qid}\tq0\t{did}\t0\t{score}\tid\n')
    # eval with -l 2 for ndcg_cut_10 
    output = subprocess.check_output(f'./trec_eval {qrel_file} {tmp_ranking_file} -m ndcg_cut.10', shell=True).decode(sys.stdout.encoding)
    ndcg_cut_10  = float(output.split('\t')[2])
    # eval with -M 10 for mrr_cut_10 
    output = subprocess.check_output(f'./trec_eval {qrel_file} {tmp_ranking_file} -m recip_rank -M 10', shell=True).decode(sys.stdout.encoding)
    mrr_cut_10  = float(output.split('\t')[2])
    # eval all other metrics
    output = subprocess.check_output(f'./trec_eval -l 2 {qrel_file} {tmp_ranking_file} -m all_trec', shell=True).decode(sys.stdout.encoding)
    metrics = {}
    for l in output.split('\n'):    
        spl = l.split('\t')
        if spl[0].strip() in save_metrics:
            metrics[spl[0].strip()] = float(spl[2])
    metrics.update({'ndcg_cut_10': ndcg_cut_10}) 
    metrics.update({'mrr_cut_10': mrr_cut_10}) 
    return metrics

def compute_metrics(p, dataset):
    dids = list()
    qids = list()
    for l in dataset:
        dids.append(l['did'])
        qids.append(l['qid'])
    predictions = p.predictions.ravel().tolist()
    metrics = eval_trec(predictions, qids, dids, dataset.qrel)
    return metrics


# Define the training arguments
training_args = TrainingArguments(
    output_dir='/tmp/',
    per_device_train_batch_size=64,
    per_device_eval_batch_size=1024,
    save_steps=10000,
    eval_steps=1000,
    max_steps=150000,
    weight_decay=0.01,
    learning_rate=2e-5,
    warmup_steps=1000,
    evaluation_strategy="steps",
    remove_unused_columns=False,
    dataloader_num_workers=4,
    bf16=True,
    #save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model='ndcg_cut_10',
    report_to='none',
)




# Create the CustomTrainer with the custom loss function
trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    #data_collator=None,  # Use the default data collator
    eval_dataset=test_dataset,
    compute_metrics=lambda p: compute_metrics(p, test_dataset),
    callbacks = [EarlyStoppingCallback(early_stopping_patience=20)]
)

# Evaluate the model
trainer.compute_metrics=lambda p: compute_metrics(p, test_dataset)
results = trainer.evaluate(eval_dataset=test_dataset)
# Print evaluation results
print(results)

