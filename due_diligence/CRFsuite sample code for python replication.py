#!/usr/bin/env python
# coding: utf-8

# ## In this code, I am running a CRFsuite model using folds created by Original paper.

# ### Running for all 5 folds for one topic

# In[1]:


from sklearn_crfsuite import CRF
import os

# Function to load data

def load_data(filenames):
    X, y = [], []
    total_lines_read = 0
    total_lines_skipped = 0

    for filename in filenames:
        with open(filename, 'r') as f:
            lines_read = 0
            lines_skipped = 0
            X_fold, y_fold = [], []
            for line in f:
                lines_read += 1
                parts = line.strip().split()
                if not parts:
                    lines_skipped += 1
                    continue
                label = parts[0]
                features = {feat.split(':')[0]: float(feat.split(':')[1]) for feat in parts[1:]}
                X_fold.append([features])
                y_fold.append(label)
            X.extend(X_fold)
            y.extend(y_fold)
            
            print(f"File: {filename}")
            print(f"  Lines read: {lines_read}")
            print(f"  Lines skipped: {lines_skipped}")
            
            total_lines_read += lines_read
            total_lines_skipped += lines_skipped

    print("Total across all folds:")
    print(f"  Total lines read: {total_lines_read}")
    print(f"  Total lines skipped: {total_lines_skipped}")
    
    return X, y


# In[2]:


# Directory containing the fold files

# dir_path = './core-tech/crf-tmp-1086/'
dir_path = './core-tech/crf-tmp-1439/'


# In[4]:


# Lists to store combined predictions and gold labels

all_preds = []
all_golds = []

for i in range(5):
    print(f"Running pass {i}")
    
    # Identify training and testing files
    fold_files = [f for f in os.listdir(dir_path) if f.startswith('fold.')]
    training_files = [f for f in fold_files if not f.endswith(f"{i}")]
    testing_file = [f for f in fold_files if f.endswith(f"{i}")][0]
    
    print("training_files", training_files)
    print("testing_file", testing_file)
    
    # Load training and testing data
    X_train, y_train = load_data([os.path.join(dir_path, f) for f in training_files])
    X_test, y_test = load_data([os.path.join(dir_path, testing_file)])
    
    # Train the model
    crf = CRF(algorithm="pa",c=0.1, max_iterations=100, pa_type=2, verbose=True)
    crf.fit(X_train, y_train)
    
    # Predict on the testing set
    y_pred = crf.predict(X_test)
    
    # Save predictions and gold labels for this fold
    all_preds.extend([label for sublist in y_pred for label in sublist])
    all_golds.extend(y_test)

# After the loop, total counts
print(f"Total rows in all_preds: {len(all_preds)}")
print(f"Total rows in all_golds: {len(all_golds)}")


# In[5]:


# # After the loop, report total counts
# print(f"Total rows in all_preds: {len(all_preds)}")
# print(f"Total rows in all_golds: {len(all_golds)}")


# In[47]:


# Save combined predictions and gold labels to files
field = '1439_test_pa'

with open(f"{field}.pred.raw", "w") as pred_file:
    pred_file.write("\n".join(all_preds))

with open(f"{field}.gold.raw", "w") as gold_file:
    gold_file.write("\n".join(all_golds))

print("Post-Processing done.")


# In[57]:


# Load predictions from saved files
with open(f"{field}.pred.raw", "r") as file:
    y_pred_flat = [line.strip() for line in file.readlines()]


# In[58]:


# Load gold from saved files
with open(f"{field}.gold.raw", "r") as file:
    y_test_flat = [line.strip() for line in file.readlines()]


# In[59]:


len(y_pred_flat), len(y_test_flat)


# In[60]:


set(y_pred_flat)


# In[61]:


set(y_test_flat)


# In[62]:


B_string_count_pred = y_pred_flat.count('B')
B_string_count_gold = y_test_flat.count('B')

print("B - count in predictions:", B_string_count_pred)
print("B - count in Actual:", B_string_count_gold)


# In[63]:


one_string_count_pred = y_pred_flat.count('1')
one_string_count_gold = y_test_flat.count('1')

print("1 - count in predictions:", one_string_count_pred)
print("1 - count in Actual:", one_string_count_gold)


# In[15]:


tp, fp, fn = 0, 0, 0
for gold, pred in zip(y_test_flat, y_pred_flat):
    # Skip empty strings
    if gold == '' or pred == '':
        continue
    gold = 0 if gold == 'B' else int(gold)
    pred = 0 if pred == 'B' else int(pred)
    
    if gold == 1 and pred == 1:
        tp += 1
    elif gold != 1 and pred == 1:
        fp += 1
    elif gold == 1 and pred != 1:
        fn += 1

# Compute metrics
eps = 1e-6
precision = tp / (tp + fp + eps)
recall = tp / (tp + fn + eps)
f1 = 2 * (precision * recall) / (precision + recall + eps)

print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")


# In[ ]:





# ### Loading the files from directory where gold and pred files are saved after running CLI experiment for the same topic

# In[36]:


# Load predictions from predictions.txt
with open(f"{dir_path}/1439_testing_CL_pa.pred.raw", "r") as file:
    y_pred_flat_1 = [line.strip() for line in file.readlines()]


# In[37]:


# Load gold from gold.txt
with open(f"{dir_path}/1439_testing_CL_pa.gold.raw", "r") as file:
    y_test_flat_1 = [line.strip() for line in file.readlines()]


# In[38]:


set(y_pred_flat_1)


# In[39]:


set(y_test_flat_1)


# In[40]:


empty_string_count = y_test_flat_1.count('')
empty_string_count


# In[54]:


B_string_count_pred_1 = y_pred_flat_1.count('B')
B_string_count_gold_1 = y_test_flat_1.count('B')

print("B - count in predictions:", B_string_count_pred_1)
print("B - count in Actual:", B_string_count_gold_1)


# In[55]:


one_string_count_pred_1 = y_pred_flat_1.count('1')
one_string_count_gold_1 = y_test_flat_1.count('1')

print("1 - count in predictions:", one_string_count_pred_1)
print("1 - count in Actual:", one_string_count_gold_1)


# In[41]:


tp, fp, fn = 0, 0, 0
for gold, pred in zip(y_test_flat_1, y_pred_flat_1):
    # Skip empty strings
    if gold == '' or pred == '':
        continue
    gold = 0 if gold == 'B' else int(gold)
    pred = 0 if pred == 'B' else int(pred)
    
    if gold == 1 and pred == 1:
        tp += 1
    elif gold != 1 and pred == 1:
        fp += 1
    elif gold == 1 and pred != 1:
        fn += 1

# Compute metrics
eps = 1e-6
precision = tp / (tp + fp + eps)
recall = tp / (tp + fn + eps)
f1 = 2 * (precision * recall) / (precision + recall + eps)

print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")


# In[ ]:




