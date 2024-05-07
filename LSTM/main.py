import os
import pandas as pd
from data_loader import load_data, preprocess_data
from glove_loader import load_glove_embeddings
from train import train_evaluate_model

def main():
    # Define paths and directories
    data_path = '../../core-tech/due_dilligence_data.csv'
    glove_embedding_path = '../../core-tech/glove.840B.300d.txt'
    model_save_dir = 'topic_models_LSTM_imbalanced'
    
    # Ensure model directory exists
    os.makedirs(model_save_dir, exist_ok=True)

    # Load and prepare data
    df = load_data(data_path)
    df['word_count'] = df['sentence'].apply(lambda x: len(str(x).split()))
    glove_embeddings = load_glove_embeddings(glove_embedding_path)
    
    # Define topic IDs for analysis
    unique_topics = [1524]  # Example of directly defining topic IDs as integers

    # Initialize metric accumulators
    sum_precision, sum_recall, sum_f1_score = 0, 0, 0
    
    # Process each topic
    for topic_id in unique_topics:
        print(f"Processing topic: {topic_id}")
        topic_data = df[df['topic_id'] == topic_id]
        
        if topic_data.empty:
            print(f"No data found for topic id: {topic_id}")
            continue
        
        print("Preprocessing data")
        X, df_filtered, max_length, tokenizer_obj, vocab_size = preprocess_data(topic_data, glove_embeddings)
        
        print("Training and evaluating model")
        precision, recall, f1_score = train_evaluate_model(
            X, df_filtered, max_length, topic_id, model_save_dir, tokenizer_obj, vocab_size, glove_embeddings)
        
        sum_precision += precision
        sum_recall += recall
        sum_f1_score += f1_score

    # Compute average metrics
    num_topics = len(unique_topics)
    avg_precision = sum_precision / num_topics
    avg_recall = sum_recall / num_topics
    avg_f1_score = sum_f1_score / num_topics

    print("Average Metrics Across All Topics:")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1_score:.4f}")

if __name__ == '__main__':
    main()
