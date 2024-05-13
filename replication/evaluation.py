import os

def labels_to_spans(labels):
    """Convert a list of labels to spans (start, end) indicating label sequences."""
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
        if label == "1" and start is None:
            start = pos
        elif label != "1" and start is not None:
            spans.append((start, pos - 1))
            start = None
    if start is not None:
        spans.append((start, len(labels) - 1))
    return spans

def load_labels_and_create_spans(file_path, output_file):
    """Load labels from a file, create spans, and write them to another file."""
    with open(file_path, 'r') as f:
        raw_data = f.readlines()

    current_sequence, all_spans = [], []
    for line in raw_data:
        label = line.strip()
        if label:
            current_sequence.append(label)
        else:
            if current_sequence:
                spans = labels_to_spans(current_sequence)
                all_spans.extend(spans)
                current_sequence = []
    if current_sequence:
        spans = labels_to_spans(current_sequence)
        all_spans.extend(spans)

    with open(output_file, 'w') as f:
        for span in all_spans:
            f.write(f"{span[0]} {span[1]}\n")

def evaluate_sentence_level(labels_gold, labels_pred):
    """Calculate sentence-level precision, recall, and F1-score."""
    tp = sum(1 for g, p in zip(labels_gold, labels_pred) if g == p == "1")
    fp = sum(1 for g, p in zip(labels_gold, labels_pred) if g != "1" and p == "1")
    fn = sum(1 for g, p in zip(labels_gold, labels_pred) if g == "1" and p != "1")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"Precision": precision, "Recall": recall, "F1": f1}

def evaluate_annotations(gold_file, pred_file):
    """Evaluate the predicted annotations against the gold standard spans."""
    gold = parse_span_file(gold_file)
    pred = parse_span_file(pred_file)
    
    tp = len([g for g in gold if any(p['start'] <= g['end'] and p['end'] >= g['start'] for p in pred)])
    fp = len([p for p in pred if not any(g['start'] <= p['end'] and g['end'] >= p['start'] for g in gold)])
    fn = len([g for g in gold if not any(p['start'] <= g['end'] and p['end'] >= p['start'] for p in pred)])

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return {"Precision": precision, "Recall": recall, "F1-Score": f1}

def parse_span_file(filename):
    """Parse a span file to extract spans."""
    spans = []
    with open(filename) as file:
        for line in file:
            start, end = map(int, line.strip().split())
            spans.append({'start': start, 'end': end})
    return spans
