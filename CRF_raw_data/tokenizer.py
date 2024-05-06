import pickle

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        return pickle.load(f)