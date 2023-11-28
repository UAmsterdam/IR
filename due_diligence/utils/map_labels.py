from datasets import Dataset


d = Dataset.load_from_disk('../data_fintech/due_diligence_kira.hf/') 


def map_(example):
    label = example['label']
    if label == 'B':
        example['label'] = 0
    else:
        example['label'] = 1
    return example

d.map(map_)
d.save_to_disk('../data_fintech/due_diligence_kira_fix_label.hf/')


