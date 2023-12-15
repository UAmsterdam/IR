#!/usr/bin/env python
# coding: utf-8

# ## Replication CODE FOR All topics 

# In[1]:


import os
import pycrfsuite
import glob
from tqdm import tqdm
from sklearn.metrics import classification_report


# In[2]:


def create_folds(directory, topid, featurization, temp_dir):
    for i in range(5):  # For each fold
        fold_filename = os.path.join(temp_dir, f"fold_{topid}.{i}")

        with open(fold_filename, "w") as fold_file:
            
            cache_files_pattern = os.path.join(directory, 'qrels', topid, f'*-{i}.cache')
#             print(cache_files_pattern)

            for cache_file in glob.glob(cache_files_pattern):
                with open(cache_file, "r") as cf:
                    for docid in cf:
                        docid = docid.strip()
                        qrels_file = os.path.join(directory, 'qrels', topid, f"{docid}.qrels")
                        docs_file = os.path.join(directory, 'docs', f"{docid}.{featurization}")


                        if os.path.exists(qrels_file) and os.path.exists(docs_file):

                            with open(qrels_file, "r") as qf, open(docs_file, "r") as df:
                                qrels_lines = qf.readlines()
                                docs_lines = df.readlines()

                                if len(qrels_lines) != len(docs_lines):
                                    print(f"Line count mismatch in {docid}: {len(qrels_lines)} vs {len(docs_lines)}")
                                    continue

                                for qrels_line, docs_line in zip(qrels_lines, docs_lines):
                                    combined_line = f"{qrels_line.strip()}\t{docs_line.strip()}\n"
                                    fold_file.write(combined_line)
                        else:
                            print(f"Missing qrels or docs file for docid: {docid}")
            fold_file.write("\n")


# In[3]:


def parse_data(file_path):
    xseq, yseq, data = [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split('\t')
                label = parts[0]
                features = parts[1].split()
                features_dict = {f'feature{i}': feature for i, feature in enumerate(features)}
                xseq.append(features_dict)
                yseq.append(label)
            else:
                if xseq and yseq:
                    data.append((xseq, yseq))
                    xseq, yseq = [], []
        if xseq and yseq:
            data.append((xseq, yseq))
    return data


# In[13]:


def train_model(training_data, i):
    print('Training model:', i)
    trainer = pycrfsuite.Trainer(verbose=True)
    for xseq, yseq in training_data:
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 0.1,
        'c2': 0.001,
        'max_iterations': 100,
        'feature.possible_transitions': True
    })

    model_file = f"training.{i}.model"
    trainer.train(model_file)
    return model_file


# In[5]:


def predict_and_write_output(tagger, testing_data):
    y_true, y_pred = [], []
    for test_seq, gold_labels in testing_data:
        prediction = tagger.tag(test_seq)
        y_pred.extend(prediction)
        y_true.extend(gold_labels)
    return y_true, y_pred


# In[6]:


def convert_labels_to_spans(y_pred):
    start, spans = 0, []
    in_span = False
    for pos, label in enumerate(y_pred):
        if label == "1" and not in_span:
            start = pos
            in_span = True
        elif label != "1" and in_span:
            spans.append((start, pos - 1))
            in_span = False
    if in_span:
        spans.append((start, len(y_pred) - 1))
    return spans


# In[7]:


def calculate_custom_metrics(y_true, y_pred):
    from collections import Counter

    label_pairs = Counter(zip(y_true, y_pred))

    tp = fp = fn = 0
    for (gold, pred), count in label_pairs.items():
        if gold == 'B':
            gold = 0
        if pred == 'B':
            pred = 0

        gold = int(gold)
        pred = int(pred)

        if gold == pred == 1:
            tp += count
        elif gold == 1 and pred != 1:
            fn += count
        elif gold != 1 and pred == 1:
            fp += count

    eps = 0.000001
    recall = tp / float(tp + fn + eps)
    precision = tp / float(tp + fp + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    return tp, fp, fn, recall, precision, f1


# In[8]:


def run_crf_tuned(field, temp_dir, base_dir):
    # Directory for temporary files specific to the topic
    temp_dir = os.path.join(base_dir, f"crf-tmp-{field}")
#     print(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Use the main base directory for accessing the actual data
    create_folds(base_dir, field, "features", temp_dir)

#     # Initialize aggregate counters
    total_tp = total_fp = total_fn = 0

    for i in tqdm(range(1)):
        training_files = [os.path.join(temp_dir, f"fold_{field}.{j}") for j in range(5) if j != i]
        print('Training folds;', training_files)

        testing_file = os.path.join(temp_dir, f"fold_{field}.{i}")
        print('Testing folds;', testing_file)

        # Initialize an empty list for training data
        training_data = []

        # Aggregate training data from each file
        for file_name in training_files:
            training_data.extend(parse_data(file_name))
            
        model_file = train_model(training_data, i)

        tagger = pycrfsuite.Tagger()
        tagger.open(model_file)
        testing_data = parse_data(testing_file)

        y_true, y_pred = predict_and_write_output(tagger, testing_data)
        tp, fp, fn, _, _, _ = calculate_custom_metrics(y_true, y_pred)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    return total_tp, total_fp, total_fn


# In[9]:


def process_all_topics(topics, base_dir):
    print("Process started for topic:", topics)
    overall_tp = overall_fp = overall_fn = 0

    for topic in topics:
        topic_dir = os.path.join(base_dir, topic)
        tp, fp, fn = run_crf_tuned(topic, topic_dir, base_dir)

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

    eps = 1e-10  # Small epsilon value to prevent division by zero

    # Calculate average metrics across all topics
    avg_recall = overall_tp / (overall_tp + overall_fn + eps)
    avg_precision = overall_tp / (overall_tp + overall_fp + eps)
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + eps)

    return avg_recall, avg_precision, avg_f1


# In[10]:


topics = ['1272', ' 1474 ', ' 1238 ', ' 1275 ', ' 1239 ', ' 1520 ', ' 1509 ', ' 1240 ', ' 1308 ', ' 1319 ',
             ' 1439 ', ' 1267 ', ' 1242 ', ' 1462 ', ' 1265 ', ' 1444 ', ' 1312 ', ' 1244 ', ' 1243 ', ' 1468 ',
             ' 1309 ', ' 1524 ', ' 1247 ', ' 1440 ', ' 1251 ', ' 1249 ', ' 1248 ', ' 1262 ', ' 1250 ', ' 1252 ',
             ' 1245 ', ' 1512 ', ' 1498 ', ' 1601 ', ' 1443 ', ' 1086 ', ' 1551 ', ' 1253 ', ' 1320 ', ' 1304 ',
             ' 1469 ', ' 1611 ', ' 1300 ', ' 1489 ', ' 1500 ', ' 1261 ', ' 1318 ', ' 1460 ', ' 1475 ', ' 1321 ']


# In[11]:


trimmed_list = [element.strip() for element in topics]


# In[14]:


base_dir = '/Users/mdwivedi/Downloads/core-tech/core'
avg_recall, avg_precision, avg_f1 = process_all_topics(trimmed_list, base_dir)
print(f"Average Recall: {avg_recall}, Precision: {avg_precision}, F1-Score: {avg_f1}")


# In[ ]:





# In[ ]:





# In[14]:


# def process_all_topics(topics, base_dir):
#     for topic in topics:
#         topic_dir = os.path.join(base_dir, topic)
# #         print(topic_dir)
#         run_crf_tuned(topic, topic_dir, base_dir)


# In[16]:


# # Example usage
# topics = ['1272', '1086']  # List of your topics
# base_dir = '/Users/mdwivedi/Downloads/core-tech/core'
# process_all_topics(topics, base_dir)


# In[ ]:





# In[ ]:




