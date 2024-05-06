from features import extract_features_and_labels, process_text_and_extract_features
from crf_model import train_crf_model, load_crf_model, predict_with_crf
import os

def train_loop(train_data, model_save_dir, topic_id, fold, tokenizer):
    ''' This function is training and saving CRF model for each fold.'''
    
    train_sentences = train_data['sentence']
    train_labels = train_data['label']
    
    print("Preprocessing start")
    
    # Extract features and labels for each training
    train_extracted = [process_text_and_extract_features(sentence, label, tokenizer) for sentence, label in zip(train_sentences, train_labels)]
    
    
    X_train = [features for features, _ in train_extracted]
    y_train = [labels for _, labels in train_extracted]

    print(f"Training model for topic {topic_id} for fold {fold}")

    fold_model_dir = os.path.join(model_save_dir, topic_id)
    
    # Create the directory if it does not exist
    os.makedirs(fold_model_dir, exist_ok=True)

    # Define the full path for the model file
    fold_model_path = os.path.join(fold_model_dir, f"{topic_id}_crf_model_{fold}.pkl")
    
    train_crf_model(X_train, y_train, fold_model_path)

    print(f"CRF model saved for topic {topic_id}")