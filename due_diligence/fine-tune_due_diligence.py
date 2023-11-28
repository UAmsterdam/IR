
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
import datasets
from transformers import AutoTokenizer
import numpy as np
from functools import partial

# loading dataset
#data = Dataset.load_from_disk("data_fintech/due_diligence_kira_unique_clean_label_fixed.hf/").shuffle(seed=42).select(range(1000000))
data = Dataset.load_from_disk("data_fintech/due_diligence_kira_unique_clean_label_fixed.hf/")
#data = Dataset.load_from_disk("data_fintech/due_diligence_kira.hf/")


# optional: filter dataset by topic
data = data.filter(lambda l: True if l['topic_id'] == '1086' else False).shuffle(seed=42)
data = data.class_encode_column("label")

# split into train and test
test_ratio = 0.001
data = data.train_test_split(test_size=test_ratio, stratify_by_column="label")

print('Num examples eval', len(data['test']))

# get number of positive and negative samples for undersampling
train_pos = data['train'].filter(lambda l: True if l['label'] == 1 else False)
train_neg = data['train'].filter(lambda l: True if l['label'] == 0 else False)

print('Num train_pos', len(train_pos), 'Num train_neg', len(train_neg))

# select number of understampled negative samples
train_neg_balanced = train_neg.shuffle(seed=42).select(range(len(train_pos)))

print('Num balanced train_neg', len(train_neg_balanced))

# merge undersampled negatives and positives into train dataset
data['train'] = datasets.concatenate_datasets([train_neg_balanced, train_pos]).shuffle(seed=42)


#loading tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# function to tokenize input
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)


# run mapping function on dataset to tokenize
tokenized_data = data.map(preprocess_function, batched=True)


model_name = "bert-base-uncased"
# load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# define data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# load eval metrics
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")


# custom function to compute f1, recall, precision
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    return {'f1': f1, 'precision': precision, 'recall': recall}



train_batch_size = 32

# define training arguments
training_args = TrainingArguments(
    output_dir="due_diligence_fine_tune",
    learning_rate=2e-5,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size*2,
    #num_train_epochs=5,
    max_steps=100000,
    weight_decay=0.01,
    evaluation_strategy='steps',
    eval_steps=1000,
    save_total_limit=5,
    fp16=True,
    run_name='due_diligence'
)

# define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# train model
trainer.train()


