# from metrics import Trec
from datasets import Dataset
import evaluate
import torch
# from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from random import randrange
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
import sys
import os
import re

# Set batch size and other relevant parameters
batch_size = 1

checkpoint_dir = 'meta-llama/Llama-2-7b-chat-hf'
dataset_path = 'data_fintech/due_diligence_kira.hf'

def format_instruction(sample):
    return f"""Sentence: {sample['sentence']}\nInstruction: You are an expert lawyer for due diligence in mergers and aquisitions. Does this sentence pose a major liability or risk for the transaction? Yes or No?\n\nAnswer:"""
    #return f"""Sentence: {sample['sentence']}\nInstruction: You are an expert lawyer for due diligence in mergers and aquisitions. Your task is to identify provisions. Does this sentence pose a major liability or risk for the transaction? yes or no?\n\nAnswer:"""


dataset = Dataset.load_from_disk(dataset_path).select(range(100))
def map_labels(example):
    if example['label'] == 'B':
        example['label'] = 0
    else:
        example['label'] = int(example['label'])
    return example


dataset = dataset.map(map_labels)

sample = dataset[randrange(len(dataset))]
prompt = format_instruction(sample)
print(sample)
print(prompt)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, padding_side='left')
no_padd = False
if not tokenizer.pad_token:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    no_padd = True
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_use_dobule_quant=True
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, device_map='auto', quantization_config=quant_config,use_flash_attention_2=True)

if no_padd:
    model.config.pretraining_tp = 1
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.model.embed_tokens.padding_idx = len(tokenizer) - 1
    model.model.embed_tokens._fill_padding_idx_with_zero()

model.config.use_cache = True



def collate_fn(batch):
    labels = [sample['label'] for sample in batch]
    sent_id = [sample['sent_id'] for sample in batch]
    topic_id = [sample['topic_id'] for sample in batch]
    instr = [format_instruction(sample) for sample in batch]  # Add prompt to each text
    instr_tokenized = tokenizer(instr, padding=True, truncation=True, return_tensors="pt", max_length=1024)
    return sent_id, topic_id, labels, instr_tokenized


output_dir = '.'

False_tokenid = tokenizer.encode('\n' + 'False', add_special_tokens=False)[-1]
true_tokenid = tokenizer.encode('\n' + 'true', add_special_tokens=False)[-1]
false_tokenid = tokenizer.encode('\n' + 'false', add_special_tokens=False)[-1]
yes_tokenid = tokenizer.encode('\n' + 'yes', add_special_tokens=False)[-1]
no_tokenid = tokenizer.encode('\n' + 'no', add_special_tokens=False)[-1]
Yes_tokenid = tokenizer.encode( '\n' + 'Yes', add_special_tokens=False)[-1]
No_tokenid = tokenizer.encode('\n' + 'No', add_special_tokens=False)[-1]
y_tokenid = tokenizer.encode('\n' + 'y', add_special_tokens=False)[-1]
n_tokenid = tokenizer.encode('\n' + 'n', add_special_tokens=False)[-1]


def get_scores(model, instr_tokenized,  print_bool):
    with torch.cuda.amp.autocast():
        scores = model.generate(**instr_tokenized.to('cuda'), max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True).scores
    scores = torch.stack(scores)
    if print_bool:
        print('max prob token', tokenizer.batch_decode(scores[:,0, :].max(1).indices), scores[:,0, :].max(1).indices.ravel().item() ,scores[:,0, :].max(1).values.ravel().item())
    scores = scores[0, :, [False_tokenid, Yes_tokenid]].float()
    prob = torch.softmax(scores, 1)
    binary_rel = (prob[:, 1] > prob[:, 0]).int()
    return binary_rel

    


# # Create a DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4)
steps = 0 
all_predictions, all_labels = list(), list()
with torch.inference_mode():
    for batch_inp in tqdm(dataloader): 
        sent_id, topic_id, labels, instr_tokenized = batch_inp
        instr_tokenized = instr_tokenized.to('cuda')
        scores = get_scores(model, instr_tokenized, steps<=10)
        all_predictions += scores.tolist()
        all_labels += labels
        steps += 1
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
accuracy_metric = evaluate.load("accuracy")
results_f1 = f1_metric.compute(predictions=all_predictions, references=all_labels)
results_precision = precision_metric.compute(predictions=all_predictions, references=all_labels)
results_accuracy = accuracy_metric.compute(predictions=all_predictions, references=all_labels)
print(results_f1, results_precision, results_accuracy)
print(all_labels)
print(all_predictions)
