import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, Embedding
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd

# Custom imports from local scripts
from callbacks import MetricsAfterEpoch, SaveModelAndTokenizerCallback, SavePredAndGoldCallback
from evaluation import load_labels_and_create_spans, evaluate_annotations
from attention import Attention

def train_evaluate_model(X, df_filtered, max_length, topic_id, model_save_dir, tokenizer_obj, vocab_size, glove_embeddings):
    """
    Trains and evaluates a text classification model using Bidirectional LSTM and Attention.
    
    Args:
        X (np.array): Input features for the model.
        df_filtered (DataFrame): Filtered pandas DataFrame containing the training data.
        max_length (int): Maximum length of sequences for padding.
        topic_id (str): Identifier for the topic under analysis.
        model_save_dir (str): Directory to save trained models and tokenizers.
        tokenizer_obj (Tokenizer): Keras Tokenizer object.
        vocab_size (int): Vocabulary size used in the model.
        glove_embeddings (dict): Preloaded GloVe embeddings.
    
    Returns:
        tuple: Returns precision, recall, and F1-score of the trained model.
    """
    embedding_dim = 300
    max_words = 20000
    word_index = tokenizer_obj.word_index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = glove_embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    print(f"Processing data for topic id: {topic_id}")
    topic_data = df_filtered[df_filtered['topic_id'] == topic_id]
    if topic_data.empty:
        print(f"No data found for topic id: {topic_id}")
        return

    sequences = tokenizer_obj.texts_to_sequences(topic_data['cleaned_sentence'])
    X_topic = pad_sequences(sequences, maxlen=max_length, padding='post')
    y_topic = topic_data['label'].values

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X_topic, y_topic, test_size=0.2, random_state=42)

    # Model definition
    input_layer = Input(shape=(max_length,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), input_length=max_length, trainable=False)(input_layer)
    bi_lstm = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
    dropout = Dropout(0.5)(bi_lstm)
    attention = Attention(max_length)(dropout)
    output = Dense(1, activation='sigmoid')(attention)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Callbacks
    metrics_callback = MetricsAfterEpoch(X_train, y_train, X_test, y_test, interval=5)
    save_model_and_tokenizer_callback = SaveModelAndTokenizerCallback(topic_id, model_save_dir, tokenizer_obj, model_save_dir, interval=5)
    save_pred_and_gold_callback = SavePredAndGoldCallback(X_test, y_test, model_save_dir, topic_id, interval=5)

    print("Model training")
    model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_test, y_test), verbose=2, callbacks=[metrics_callback, save_model_and_tokenizer_callback, save_pred_and_gold_callback])

    print("Model prediction")
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Evaluation
    cm = confusion_matrix(y_test, y_pred_binary)
    report = classification_report(y_test, y_pred_binary, target_names=['Class 0', 'Class 1'])
    TP = cm[1, 1]
    FP = cm[0, 1]
    TN = cm[0, 0]
    FN = cm[1, 0]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Topic: {topic_id}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
    
    y_pred_str = [str(pred) for pred in np.concatenate(y_pred_binary)]
    y_test_str = [str(label) for label in y_test]

    pred_file_path = os.path.join(model_save_dir, str(topic_id), f"{topic_id}.pred.raw")
    gold_file_path = os.path.join(model_save_dir, str(topic_id), f"{topic_id}.gold.raw")

    # Ensure the directory exists before attempting to write the files
    os.makedirs(os.path.dirname(pred_file_path), exist_ok=True)

    with open(pred_file_path, "w") as pred_file, open(gold_file_path, "w") as gold_file:
        pred_file.write("\n".join(y_pred_str))
        gold_file.write("\n".join(y_test_str))

    load_labels_and_create_spans(pred_file_path, os.path.join(model_save_dir, str(topic_id), f"{topic_id}.pred.span"))
    load_labels_and_create_spans(gold_file_path, os.path.join(model_save_dir, str(topic_id), f"{topic_id}.gold.span"))

    metrics = evaluate_annotations(os.path.join(model_save_dir, str(topic_id), f"{topic_id}.gold.span"), os.path.join(model_save_dir, str(topic_id), f"{topic_id}.pred.span"))
    print(metrics)

    return precision, recall, f1_score

