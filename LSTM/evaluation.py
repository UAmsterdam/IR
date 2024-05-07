from itertools import chain
import os
import pandas as pd

def labels_to_spans(labels):
    """
    Converts a list of labels ('1' or '0') into spans.
    
    Args:
    - labels: List of labels ('1' or '0') for tokens.

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
    """
    Loads labels from a file, creates spans, and writes them to an output file.
    
    Args:
    - file_path: Path to the file containing labels.
    - output_file: Path to the output file where spans will be written.
    """
    with open(file_path, 'r') as file:
        raw_data = file.readlines()

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
    
    with open(output_file, 'w') as file:
        for span in all_spans:
            file.write(f"{span[0]} {span[1]}\n")

    
    # Write the spans to the output file
    with open(output_file, 'w') as f:
        for span in all_spans:
            f.write(f"{span[0]} {span[1]}\n")

def parse_span_file(filename):
    """
    Parses a span file to extract spans as dictionaries with 'start' and 'end'.
    
    Args:
    - filename: Path to the span file.
    
    Returns:
    - List of dictionaries representing spans.
    """
    spans = []
    with open(filename) as file:
        for line in file:
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
    
    return {"TP": tp, "FP": fp, "FN": fn, "Recall": recall, "Precision": precision, "F1-Score": f1}