from datasets import Dataset
from collections import defaultdict
from tqdm import tqdm
import pickle
d = Dataset.load_from_disk('data_fintech/due_diligence_kira.hf/')

print(d)
#['sentence', 'label', 'index', 'sent_id', 'topic_id'],
new = defaultdict(lambda: defaultdict(list))
for el in tqdm(d):
    sent_id = el['sent_id']
    doc_id = sent_id.split('_')[0]
    new[doc_id]['document'].append(el['sentence'])
    new[doc_id]['topic_id'].append(el['topic_id'])
    new[doc_id]['label'].append(el['label'])
pickle.dump(new, open('new.pkl', 'wb'))
