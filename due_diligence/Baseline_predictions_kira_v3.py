#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pycrfsuite
import shutil
from pathlib import Path
from tqdm import tqdm
import glob
from pathlib import Path
from sklearn.metrics import classification_report

def create_folds(directory, topid, featurization):
    for i in range(5):  # For each fold
        fold_filename = f"fold.{i}"
        with open(fold_filename, "w") as fold_file:
            cache_files_pattern = os.path.join(directory, 'qrels', topid, f'*-{i}.cache')

            for cache_file in glob.glob(cache_files_pattern):
                with open(cache_file, "r") as cf:
                    for docid in cf:
                        docid = docid.strip()
                        qrels_file = os.path.join(directory, 'qrels', topid, f"{docid}.qrels")
                        docs_file = os.path.join(directory, 'docs', f"{docid}.{featurization}")

                        if os.path.exists(qrels_file) and os.path.exists(docs_file):
                            with open(qrels_file, "r") as qf, open(docs_file, "r") as df:
                                # Read all lines for each file
                                qrels_lines = qf.readlines()
                                docs_lines = df.readlines()

                                # Check if the number of lines matches in both files
                                if len(qrels_lines) != len(docs_lines):
                                    print(f"Line count mismatch in {docid}: {len(qrels_lines)} vs {len(docs_lines)}")
                                    continue

                                for qrels_line, docs_line in zip(qrels_lines, docs_lines):
                                    qrels_line = qrels_line.strip()
                                    docs_line = docs_line.strip()

                                    combined_line = f"{qrels_line}\t{docs_line}\n"
                                    fold_file.write(combined_line)


def convert_labels_to_spans(file_content):
    start = 0
    pos = 0
    in_span = False
    spans = []

    for line in file_content:
        line = line[0]
        if line == "1" and not in_span:
            start = pos
            in_span = True
        elif line != "1" and in_span:
            in_span = False
            spans.append((start, pos - 1))
            start = 0
        elif line == "\n" and in_span:
            in_span = False
            spans.append((start, pos - 1))
            start = 0
        pos += 1

        # In case there are 1's at the end of a fold
        if line == "1" and in_span:
            spans.append((start, pos - 1))

    return spans



def select_train_test_files(fold_index):
    print("In the training and testing folder")
    all_files = os.listdir('.')
    training_files = [f for f in all_files if f.startswith("fold.") and not f.endswith(f"{fold_index}")]
    testing_files = [f for f in all_files if f.startswith("fold.") and f.endswith(f"{fold_index}")]
    return training_files, testing_files


def parse_training_data(file_path):
    xseq = []
    yseq = []
    data = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()  # Splitting the line into parts
                label = parts[0]      # The first part is the label
                features = parts[1:]  # The rest are features

                # Construct a feature dictionary for this line
                features_dict = {f'feature{i}': feature for i, feature in enumerate(features)}
                
                xseq.append(features_dict)
                yseq.append(label)
            else:
                # Blank line indicates the end of a sequence
                if xseq and yseq:
                    data.append((xseq, yseq))
                    xseq = []
                    yseq = []

    # Add the last sequence if the file doesn't end with a blank line
    if xseq and yseq:
        data.append((xseq, yseq))

    return data

def parse_testing_data(file_path):
    xseq = []
    yseq = []
    test_data = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split()  # Splitting the line into parts
                label = parts[0]      # The first part is the label
                features = parts[1:]  # The rest are features

                # Construct a feature dictionary for this line
                features_dict = {f'feature{i}': feature for i, feature in enumerate(features)}
                
                xseq.append(features_dict)
                yseq.append(label)
            else:
                # Blank line indicates the end of a sequence
                if xseq and yseq:
                    test_data.append((xseq, yseq))
                    xseq = []
                    yseq = []

    # Add the last sequence if the file doesn't end with a blank line
    if xseq and yseq:
        test_data.append((xseq, yseq))

    return test_data


def load_training_data(training_files):
    training_data = []
    for file_name in training_files:
        training_data.extend(parse_training_data(file_name))
    return training_data


def train_model(training_data, i):
    print("Training model-- started")
    trainer = pycrfsuite.Trainer(verbose=True)
    for xseq, yseq in training_data:
        trainer.append(xseq, yseq)
        
    print("Xseq data:",xseq[0:5])
    print("Yseq data:",yseq[0:5])

    trainer.set_params({
        'c1': 0.1, 'c2': 0.01, 'max_iterations': 100,
        'feature.possible_transitions': True
    })

    model_file = f"training.{i}.model"
    trainer.train(model_file)
    print("Model training-- end")
    return model_file


def load_testing_data(testing_files):
    testing_data = []
    for file_name in testing_files:
        testing_data.extend(parse_testing_data(file_name))
    return testing_data




def predict_and_write_output(tagger, testing_data):
    print("Predictions")
    y_true = []
    y_pred = []

    for test_seq, gold_labels in testing_data:
        print("test sequence:", test_seq[:5])
        print("gold labels:", gold_labels[:5])
        prediction = tagger.tag([{f'feature{i}': item[f'feature{i}'] for i in range(len(item))} for item in test_seq])
        y_pred.extend(prediction)
        y_true.extend(gold_labels)

    return y_true, y_pred


def generate_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print(report)

def run_crf_tuned(field, dir):
    temp_dir = f"crf-tmp-{field}"
    os.makedirs(temp_dir, exist_ok=True)
    os.chdir(temp_dir)

    create_folds(dir, field, "features")

    for i in tqdm(range(5)):
        print(f"Running pass {i}")

        # Select training and testing files
        training_files, testing_files = select_train_test_files(i)
        print("Training files:",training_files)
        print("Testing files:",testing_files)
        
        training_data = load_training_data(training_files)
        model_file = train_model(training_data, i)

        tagger = pycrfsuite.Tagger()
        tagger.open(model_file)
        testing_data = load_testing_data(testing_files)

        y_true, y_pred = predict_and_write_output(tagger, testing_data)
        generate_classification_report(y_true, y_pred)


# Running code
field = '1086'
dir = '/Users/mdwivedi/Downloads/core-tech/core'
run_crf_tuned(field, dir)