#!/usr/bin/env python
# coding: utf-8

# # Finding most similar topics to train and test

# ## Using clustering on topics based on titles and description

from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


# Load and parse the XML file
tree = ET.parse('kira-topics.xml')
root = tree.getroot()


# Extract topic information
topics = []
for topic in root.findall('.//topic'):
    topid = topic.find('topid').text
    title = topic.find('title').text
    description = topic.find('description').text
    topics.append({'topid': topid, 'title': title, 'description': description})
# topics

# Convert to DataFrame
df = pd.DataFrame(topics)


# Combine title and description for vectorization
df['text'] = df['title'] + " " + df['description']


# Vectorize the text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(X)


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Calculate WCSS for a range of cluster numbers
wcss = []
for i in range(1, 11):  # Adjust the range as needed
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting
plt.figure(figsize=(5, 3))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of clusters')
# Within cluster sum of squares
plt.ylabel('WCSS')  
plt.show()


# Cluster topics using K-Means
kmeans = KMeans(n_clusters=8)  # Adjust the number of clusters as needed
df['cluster'] = kmeans.fit_predict(X)


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Let's use PCA to reduce the dimensionality of the TF-IDF vectors for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

# Plotting the clusters
plt.figure(figsize=(5, 8))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=df['cluster'], cmap='viridis', marker='o')

plt.title('Topic Clusters (Reduced to 2D using PCA)')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.colorbar(label='Cluster ID')
plt.show()


# Analyze clusters to pick topics for training
for cluster_num in sorted(df['cluster'].unique()):
    print(f"Cluster {cluster_num}:")
    print(df[df['cluster'] == cluster_num]['topid'])
    print()

exp_name = 'models/ft_due_diligence_kira'
run_name = 'topics_1439'
#model_name = 'nlpaueb/legal-bert-base-uncased'
model_name = "Support Vector Classifier"
dataset_name = 'due_dilligence_kira.hf'


# loading dataset
data = Dataset.load_from_disk(dataset_name)

# Define training and testing topics
train_topics = ['1086', '1512'] 
test_topics = ['1520', '1249']

# Split the dataset into training and testing based on topics
train_data = data.filter(lambda example: example['topic_id'] in train_topics)
test_data = data.filter(lambda example: example['topic_id'] in test_topics)

# Convert to pandas DataFrame
train_df = pd.DataFrame({'text': train_data['sentence'], 'label': train_data['label']})
train_df.shape



test_df = pd.DataFrame({'text': test_data['sentence'], 'label': test_data['label']})
test_df.shape


# Vector Extraction with TF-IDF
print("Vectorizing")
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_df['text'])
y_train = train_df['label']


X_test = vectorizer.transform(test_df['text'])
y_test = test_df['label']


# Handle class imbalance using SMOTE
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# Train the Support Vector Classifier
svc = LinearSVC()
svc.fit(X_train_smote, y_train_smote)


# Predict and Evaluate on the test set
y_pred = svc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


