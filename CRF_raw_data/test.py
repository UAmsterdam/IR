import os
from features import extract_features_and_labels, process_text_and_extract_features
from crf_model import train_crf_model, load_crf_model, predict_with_crf
from evaluation import sentence_level_results, load_labels_and_create_spans, evaluate_annotations, evaluate_model_predictions

def test_loop(test_unbalanced, model_save_dir, folder, fold, results_df, tokenizer):
    
    import pandas as pd
    import os
    
    fold_model_path = os.path.join(model_save_dir, folder, f"{folder}_crf_model_{fold}.pkl")
    
    if os.path.exists(fold_model_path):
        crf_model = load_crf_model(fold_model_path)

        # Extract features and predict
        test_sentences = test_unbalanced['sentence']
        test_labels = test_unbalanced['label']

        test_extracted = [process_text_and_extract_features(sentence, label, tokenizer) for sentence, label in zip(test_sentences, test_labels)]
        
        X_test = [features for features, _ in test_extracted]
        y_test = [labels for _, labels in test_extracted]

        ## Predict
        y_pred = predict_with_crf(crf_model, X_test)
        results_df = evaluate_model_predictions(y_test, y_pred, fold, results_df, model_save_dir, folder)

    else:
        print(f"Model file not found: {fold_model_path}")
    
    return results_df
