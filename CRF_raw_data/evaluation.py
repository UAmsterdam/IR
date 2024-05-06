from itertools import chain
import os
import pandas as pd

## Sentence Level evaluation

def sentence_level_results(true_labels_flat, predicted_labels_flat):
    
    tp, fp, fn = 0, 0, 0
    for gold, pred in zip(true_labels_flat, predicted_labels_flat):
        # Skip empty strings
        if gold == '' or pred == '':
            continue
        gold = 0 if gold == 'B' else int(gold)
        pred = 0 if pred == 'B' else int(pred)

        if gold == 1 and pred == 1:
            tp += 1
        elif gold != 1 and pred == 1:
            fp += 1
        elif gold == 1 and pred != 1:
            fn += 1

    # Compute metrics
    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    
    return precision, recall, f1
##################################### Annotation Level Evaluation ################################
def labels_to_spans(labels):
    """
    Converts a list of labels ('1' or 'B') into spans.
    
    Args:
    - labels: List of labels ('1' or 'B') for tokens.

    Returns:
    - A list of spans represented as tuples (start, end).
    """
    start = None
    spans = []
    for pos, label in enumerate(labels):
        if label == "1" and start is None:  # Start of a new span
            start = pos
        elif label != "1" and start is not None:  # End of the current span
            spans.append((start, pos - 1))
            start = None
    if start is not None:  # If a span extends to the end of the sequence
        spans.append((start, len(labels) - 1))
        
    print(len(spans))
        
    return spans

def load_labels_and_create_spans(file_path, output_file):
    with open(file_path, 'r') as f:
        raw_data = f.readlines()
        
    # Initialize variables
    current_sequence = []
    all_spans = []
    
    for line in raw_data:
        label = line.strip()
        if label:  # If the line is not empty, add the label to the current sequence
            current_sequence.append(label)
        else:  # If the line is empty, process the current sequence and reset it
            if current_sequence:  # Check if the current sequence is not empty
                spans = labels_to_spans(current_sequence)
                all_spans.extend(spans)
                current_sequence = []  # Reset the sequence for the next block

    # Process the last sequence if the file doesn't end with a blank line
    if current_sequence:
        spans = labels_to_spans(current_sequence)
        all_spans.extend(spans)
    
    # Write the spans to the output file
    with open(output_file, 'w') as f:
        for span in all_spans:
            f.write(f"{span[0]} {span[1]}\n")

def parse_span_file(filename):
    """Parses a span file to extract spans."""
    spans = []
    with open(filename) as fil:
        for line in fil:
            start, end = map(int, line.strip().split())
            spans.append({'start': start, 'end': 1 + end})
    return spans

def span_overlaps(A, B):
    """Checks if two spans overlap."""
    return not ((A['end'] <= B['start']) or (A['start'] >= B['end']))

def overlapping_spans(A, B):
    """Finds overlapping spans between two lists of spans."""
    return [a for a in A if any(span_overlaps(a, b) for b in B)]

def non_overlapping_spans(A, B):
    """Finds spans in A that do not overlap with any span in B."""
    return [a for a in A if not any(span_overlaps(a, b) for b in B)]

def evaluate_annotations(gold_file, pred_file):
    """Evaluates the predicted annotations against the gold standard."""
    gold = parse_span_file(gold_file)
    pred = parse_span_file(pred_file)
    
    eps = 0.000001
    tp = len(overlapping_spans(gold, pred))
    fp = len(non_overlapping_spans(pred, gold))
    fn = len(non_overlapping_spans(gold, pred))
    
    recall = tp / float(tp + fn + eps)
    precision = tp / float(tp + fp + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    
    return {"TP": tp, "FP": fp, "FN": fn, "Precision": precision, "Recall": recall, "F1-Score": f1}

############################################### Evaluation of all metrics ###########################################################

def evaluate_model_predictions(y_test, y_pred, fold, results_df, model_save_dir, folder):
    """Evaluate the model predictions using appropriate metrics.
    -- Setence level 
    -- Annotation level
    """
    

    topic_id = folder
    y_test_flat = list(chain.from_iterable(y_test))
    y_pred_flat = list(chain.from_iterable(y_pred))
    
    # Save the combined predictions and gold labels for the topic
    pred_file_path = os.path.join(model_save_dir, topic_id, f"{topic_id}_{fold}.pred.raw")
    gold_file_path = os.path.join(model_save_dir, topic_id, f"{topic_id}_{fold}.gold.raw")

    with open(pred_file_path, "w") as pred_file, open(gold_file_path, "w") as gold_file:
        pred_file.write("\n".join(y_test_flat))
        gold_file.write("\n".join(y_pred_flat))
    
    precision, recall, f1_score = sentence_level_results(y_test_flat, y_pred_flat)
    print(f"Avg Precision: {precision:.4f}, Avg Recall: {recall:.4f}, Avg F1-Score: {f1_score:.4f}")
    
    # Create spans and save to .span files
    load_labels_and_create_spans(pred_file_path, os.path.join(model_save_dir, topic_id, f"{topic_id}_{fold}.pred.span"))
    load_labels_and_create_spans(gold_file_path, os.path.join(model_save_dir, topic_id, f"{topic_id}_{fold}.gold.span"))

    # Evaluate the annotations
    metrics = evaluate_annotations(os.path.join(model_save_dir, topic_id, f"{topic_id}_{fold}.gold.span"), os.path.join(model_save_dir, topic_id, f"{topic_id}_{fold}.pred.span"))
    print(metrics)
    
    new_row = pd.DataFrame([{
        "Topic ID": topic_id,
        "Fold": fold,
        "Sentence Precision": precision,
        "Sentence Recall": recall,
        "Sentence F1-Score": f1_score,
        "Annotation TP": metrics["TP"],
        "Annotation FP": metrics["FP"],
        "Annotation FN": metrics["FN"],
        "Annotation Precision": metrics["Precision"],
        "Annotation Recall": metrics["Recall"],
        "Annotation F1-Score": metrics["F1-Score"],
    }], columns=results_df.columns)

    results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    return results_df

