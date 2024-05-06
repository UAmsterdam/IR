import nltk
from nltk.util import ngrams
from nltk import sent_tokenize, word_tokenize
from tokenizer import load_tokenizer

def extract_features_and_labels(sentence, label):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    
    # Generate n-grams (unigrams, bigrams, trigrams)
    unigrams = list(ngrams(tokens, 1))
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))
    
    # Generate features for each token
    token_features = []
    token_labels = []
    for i, token in enumerate(tokens):
        features = {
            'token': token,
            'lower': token.lower(),
            'is_first': i == 0,
            'is_last': i == len(tokens) - 1,
            'is_capitalized': token[0].isupper(),
            'is_all_caps': token.isupper(),
            'is_all_lower': token.islower(),
            'prefix-1': token[0],
            'prefix-2': token[:2],
            'prefix-3': token[:3],
            'suffix-1': token[-1],
            'suffix-2': token[-2:],
            'suffix-3': token[-3:],
            'prev_token': '' if i == 0 else tokens[i - 1],
            'next_token': '' if i == len(tokens) - 1 else tokens[i + 1],
            'is_numeric': token.isdigit(),
            'unigram': unigrams[i][0] if i < len(unigrams) else '',
            'bigram': ' '.join(bigrams[i]) if i < len(bigrams) else '',
            'trigram': ' '.join(trigrams[i]) if i < len(trigrams) else '',
        }
        token_features.append(features)
        token_labels.append(label)
        
    return token_features, token_labels

def process_text_and_extract_features(text, label, tokenizer):
    # Segment the text into sentences using the custom tokenizer
    sentences = tokenizer.tokenize(text)
    
    # Extract features for each sentence
    all_features = []
    all_labels = []
    for sentence in sentences:
        sentence_features, sentence_labels = extract_features_and_labels(sentence, label)
        all_features.extend(sentence_features)
        all_labels.extend(sentence_labels)
    
    return all_features, all_labels