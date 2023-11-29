
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
import datasets
from transformers import AutoTokenizer
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# init

exp_name = 'models/ft_due_diligence_kira'
run_name = 'topics_1439'
#model_name = 'nlpaueb/legal-bert-base-uncased'
model_name = "bert-base-uncased"
dataset_name = 'data_fintech/due_diligence_kira.hf'


# loading dataset
data = Dataset.load_from_disk(dataset_name)


# optional: filter dataset by topic
data = data.filter(lambda l: True if l['topic_id'] == '1439' else False, desc='Filtering dataset for topic.').shuffle(seed=42)


labels = data['label']
texts = data['sentence']

print('vectorizing')
# Convert text to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # Adjust max_features as needed
X = tfidf_vectorizer.fit_transform(texts)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#from sklearn.preprocessing import MinMaxScaler
#scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
#X_train = scaling.transform(X_train)
#X_test = scaling.transform(X_test)


# Create SVC model
svc_model = LinearSVC(verbose=1)
#svc_model = SVC(verbose=1)

print('training')
# Train the model
svc_model.fit(X_train, y_train)



print('testing')
# Make predictions on the test set
y_pred = svc_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print classification report
print(classification_report(y_test, y_pred))

