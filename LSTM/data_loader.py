import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure you have a script or module to load GloVe embeddings
from glove_loader import load_glove_embeddings

def load_data(file_path):
    """Load data from a CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def preprocess_data(df, glove_embeddings):
    """Preprocess data by tokenizing, cleaning, and encoding sentences.

    Args:
        df (DataFrame): The data frame containing the text data.
        glove_embeddings (dict): A dictionary with word embeddings.

    Returns:
        tuple: A tuple containing the tokenized text data, the filtered DataFrame, 
               the maximum sequence length, tokenizer object, and vocabulary size.
    """
    # Define mapping for labels
    mapping = {'B': 0, '1': 1}
    df['label'] = df['label'].map(mapping)

    # Filter data based on word count criteria
    df_filtered = df[df['word_count'].between(5, 20, inclusive='right')]

    # Remove stopwords and apply lemmatization
    stop_words = set(stopwords.words('english'))
    lmtzr = WordNetLemmatizer()

    def clean_text(text):
        tokens = word_tokenize(text.lower())
        stripped = [w.translate(str.maketrans('', '', string.punctuation)) for w in tokens]
        words = [w for w in stripped if w.isalpha() and w not in stop_words]
        return ' '.join([lmtzr.lemmatize(w) for w in words])

    df_filtered['cleaned_sentence'] = df_filtered['sentence'].apply(clean_text)

    # Tokenize text
    tokenizer_obj = Tokenizer(num_words=20000)
    tokenizer_obj.fit_on_texts(df_filtered['cleaned_sentence'])
    sequences = tokenizer_obj.texts_to_sequences(df_filtered['cleaned_sentence'])

    # Prepare embedding matrix
    word_index = tokenizer_obj.word_index
    vocab_size = min(len(word_index) + 1, 20000)
    max_length = max(len(x) for x in sequences)
    embedding_dim = 300
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in word_index.items():
        if i < 20000:
            embedding_vector = glove_embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    X = pad_sequences(sequences, maxlen=max_length, padding='post')

    print("vocab_size:", vocab_size)
    print("max_length:", max_length)

    return X, df_filtered, max_length, tokenizer_obj, vocab_size

