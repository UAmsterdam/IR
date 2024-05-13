import os
from sklearn_crfsuite import CRF

from evaluate import *
from data_loader import load_data

def train_and_predict_for_topic(topic_id, dir_path):
    """Train a CRF model and predict results for a given topic using 5-fold cross-validation."""
    fold_files = [f for f in os.listdir(dir_path) if f.startswith('fold.')]
    all_preds, all_golds = [], []

    for i in range(5):
        training_files = [f for f in fold_files if not f.endswith(f"{i}")]
        testing_file = [f for f in fold_files if f.endswith(f"{i}")][0]

        X_train, y_train = load_data([os.path.join(dir_path, f) for f in training_files])
        X_test, y_test = load_data([os.path.join(dir_path, testing_file)])

        crf = CRF(algorithm="pa", c=0.1, max_iterations=100, pa_type=2, verbose=True)
        crf.fit(X_train, y_train)
        y_pred = crf.predict(X_test)

        all_preds.extend(y_pred)
        all_golds.extend(y_test)

    pred_file_path = os.path.join(dir_path, f"{topic_id}.pred.raw")
    gold_file_path = os.path.join(dir_path, f"{topic_id}.gold.raw")
    with open(pred_file_path, "w") as pred_file, open(gold_file_path, "w") as gold_file:
        for pred in all_preds:
            pred_file.write("\n".join(pred) + "\n\n")
        for gold in all_golds:
            gold_file.write("\n".join(gold) + "\n\n")

    # Flatten lists for sentence-level evaluation
    flat_preds = [label for sublist in all_preds for label in sublist]
    flat_golds = [label for sublist in all_golds for label in sublist]
    sentence_metrics = evaluate_sentence_level(flat_golds, flat_preds)

    load_labels_and_create_spans(pred_file_path, os.path.join(dir_path, f"{topic_id}.pred.span"))
    load_labels_and_create_spans(gold_file_path, os.path.join(dir_path, f"{topic_id}.gold.span"))

    # annotation_metrics = evaluate_annotations(os.path.join(dir_path, f"{topic_id}.gold.span"), os.path.join(dir_path, f"{topic_id}.pred.span"))

    return sentence_metrics