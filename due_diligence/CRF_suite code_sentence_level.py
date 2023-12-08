#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pycrfsuite
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import re
import string
from collections import Counter
import random
from sklearn.model_selection import KFold
from tqdm import tqdm


# In[2]:


import sklearn_crfsuite
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from highlight_text import HighlightText, ax_text, fig_text
import random

from datasets import Dataset
import datasets


# In[3]:


exp_name = 'models/ft_due_diligence_kira'
run_name = 'topics_1439'
#model_name = 'nlpaueb/legal-bert-base-uncased'
model_name = "Support Vector Classifier"
dataset_name = 'due_dilligence_kira.hf'


# loading dataset
data = Dataset.load_from_disk(dataset_name)


# In[4]:


# optional: filter dataset by topic
data = data.filter(lambda l: True if l['topic_id'] == '1439' else False, desc='Filtering dataset for topic.').shuffle(seed=42)


# In[5]:


labels = data['label']
texts = data['sentence']


# In[6]:


# training = ' . '.join(data['sentence'])


# In[7]:


len(texts)


# In[8]:


# def pos_tags(document):
#         sentences = nltk.sent_tokenize(document) 
#         sentences = [nltk.word_tokenize(sent) for sent in sentences]
#         sentences = [nltk.pos_tag(sent) for sent in sentences]
#         return sentences
# training = pos_tags(training)


# In[9]:


# training[:5]


# In[10]:


def clean_text(text):
    """ 
    Basic text cleaning: 
    - Lowercasing 
    - Removing special characters except some punctuation
    """
    # Keeping periods, commas, and hyphens for now; you can adjust this list
    allowed_punctuation = ".,-"
    # Create a translation table for str.translate
    # that maps each special character to None
    # We'll keep the allowed punctuation and alphanumeric characters
    remove_chars = string.punctuation + string.whitespace
    remove_chars = remove_chars.translate({ord(c): None for c in allowed_punctuation})
    translation_table = str.maketrans('', '', remove_chars)

    text = text.translate(translation_table)
    return text.lower()


# In[11]:


def word2features(sentence, i):
    """ Function to extract features from a word in a sentence. """
    word = sentence[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],  # Last three characters
        'word[-2:]': word[-2:],  # Last two characters
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }

    if i > 0:
        word1 = sentence[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    if i < len(sentence)-1:
        word1 = sentence[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True  # End of sentence

    return features


# In[12]:


def extract_features(sentence):
    """ Extract features from a sentence. """
    sentence = [clean_text(word) for word in word_tokenize(sentence)]
    return [word2features(sentence, i) for i in range(len(sentence))]


# In[13]:


def prepare_data(sentences, labels):
    """ Prepare data for CRF training. """
    X = []
    Y = []
    for sentence, label in zip(sentences, labels):
        features = extract_features(sentence)
        labels = [label] * len(features)
        X.append(features)
        Y.append(labels)
    return X, Y


# In[14]:


def undersample_sequences(sentences, labels, majority_class):
    # Pair each sentence with its label
    paired = list(zip(sentences, labels))

    # Separate the majority and minority class instances
    majority_class_instances = [pair for pair in paired if pair[1] == majority_class]
    minority_class_instances = [pair for pair in paired if pair[1] != majority_class]

    # Randomly select instances from the majority class
    # The number selected is equal to the number of instances in the minority class
    k=12
    random.shuffle(majority_class_instances)
    majority_class_instances = majority_class_instances[:k*(len(minority_class_instances))]

    # Combine the downsampled majority class instances with all minority class instances
    balanced_data = majority_class_instances + minority_class_instances

    # Shuffle the combined data to mix the class instances
    random.shuffle(balanced_data)

    # Unzip the sentences and labels
    undersampled_sentences, undersampled_labels = zip(*balanced_data)
    return list(undersampled_sentences), list(undersampled_labels)


# In[15]:


#  Assuming sentences is a list of sentences and labels is a list of corresponding labels
majority_class = "B"
undersampled_sentences, undersampled_labels = undersample_sequences(texts, labels, majority_class)


# In[16]:


len(undersampled_labels)


# In[29]:


# Split data into training and test sets
# sentences_train, sentences_test, labels_train, labels_test = train_test_split(undersampled_sentences, undersampled_labels, test_size=0.3, random_state=48)
sentences_train, sentences_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=48)


# In[30]:


# Prepare training and test data
X_train, y_train = prepare_data(sentences_train, labels_train)
X_test, y_test = prepare_data(sentences_test, labels_test)


# ## K-fold cross validation

# In[31]:


# # Prepare data
# X, y = prepare_data(sentences_train, labels_train)

# # Define k-fold cross-validation
# kf = KFold(n_splits=5)  # Number of folds

# for train_index, test_index in tqdm(kf.split(X)):
#     X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
#     y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

#     # Train the model
#     trainer = pycrfsuite.Trainer(verbose=False)
#     for xseq, yseq in zip(X_train, y_train):
#         trainer.append(xseq, yseq)

#     trainer.set_params({
#         'c1': 0.1, 'c2': 0.01, 'max_iterations': 200,
#         'feature.possible_transitions': True
#     })

#     model_filename = 'temp_crf.model'
#     trainer.train(model_filename)

#     # Evaluate the model
#     tagger = pycrfsuite.Tagger()
#     tagger.open(model_filename)

#     y_pred = [tagger.tag(xseq) for xseq in X_test]
#     y_true = [label for sublist in y_test for label in sublist]
#     y_pred_flat = [label for sublist in y_pred for label in sublist]

#     # Print classification report for each fold
#     print(classification_report(y_true, y_pred_flat))


# In[32]:


len([x for x in undersampled_labels if x =="1"])


# In[33]:


X_train


# In[34]:


# Train the model
trainer = pycrfsuite.Trainer(verbose=True)


# In[35]:


# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)


# In[36]:


# Set training parameters
trainer.set_params({
    'c1': 0.1,  # coefficient for L1 penalty
    'c2': 0.05,  # coefficient for L2 penalty
    'max_iterations': 300,
    'feature.possible_transitions': True
})


# In[37]:


# Train the model
model_filename = 'crf.model_1439'
trainer.train(model_filename)


# In[38]:


# Evaluate the model
tagger = pycrfsuite.Tagger()
tagger.open(model_filename)


# In[39]:


# Flatten the test dataset for evaluation
y_pred = [tagger.tag(xseq) for xseq in X_test]
y_true = [label for sublist in y_test for label in sublist]
y_pred_flat = [label for sublist in y_pred for label in sublist]


# In[40]:


# Classification Report
print("F1 Score:",f1_score(y_true, y_pred_flat, average='weighted'))
print(classification_report(y_true, y_pred_flat))


# ## Hyperparameter tuning experiment

# In[19]:


# Hyperparameter Grid
c1_values = [0.1, 0.5, 1.0]
c2_values = [0.01, 0.05, 0.1]
max_iterations_values = [50, 100, 200]

best_score = 0
best_params = {}


# In[24]:


for c1 in tqdm(c1_values):
    for c2 in tqdm(c2_values):
        for max_iter in tqdm(max_iterations_values):
            trainer = pycrfsuite.Trainer(verbose=False)
            print("In the data loop")
            for xseq, yseq in zip(X_train, y_train):
                trainer.append(xseq, yseq)

            trainer.set_params({
                'c1': c1,
                'c2': c2,
                'max_iterations': max_iter,
                'feature.possible_transitions': True
            })
            
            print("Training the model")

            model_filename = 'temp_crf.model'
            trainer.train(model_filename)

            tagger = pycrfsuite.Tagger()
            tagger.open(model_filename)
            
            print("predictions")

            y_pred = [tagger.tag(xseq) for xseq in X_test]
            y_true = [label for sublist in y_test for label in sublist]
            y_pred_flat = [label for sublist in y_pred for label in sublist]

            score = f1_score(y_true, y_pred_flat, average='weighted')

            if score > best_score:
                best_score = score
                best_params = {'c1': c1, 'c2': c2, 'max_iterations': max_iter}


# In[25]:


print("Best F1 Score:", best_score)
print("Best Hyperparameters:", best_params)


# In[ ]:




