import pickle
from datasets import Dataset


qrel_topic = pickle.load(open('data_fintech/due_diligence_kira_qrel_1439.pkl', 'rb'))

sent_ids  = set([ el[0] for el in qrel_topic])

data = Dataset.load_from_disk('data_fintech/due_diligence_kira.hf')

data = data.filter(lambda el: el['topic_id'] == '1439' and el['sent_id'] in sent_ids)

data.save_to_disk('due_diligence_kira_topic_1439.hf')

