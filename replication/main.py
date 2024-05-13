import os

# Configurable base directory for data files
BASE_DIR = './data'

def main():
    unique_topics = ['1272', '1474', '1238', '1275', '1239', '1520', '1509', '1240', '1308', '1319',
                     '1439', '1267', '1242', '1462', '1265', '1444', '1312', '1244', '1243', '1468',
                     '1309', '1524', '1247', '1440', '1251', '1249', '1248', '1262', '1250', '1252',
                     '1245', '1512', '1498', '1601', '1443', '1086', '1551', '1253', '1320', '1304',
                     '1469', '1611', '1300', '1489', '1500', '1261', '1318', '1460', '1475', '1321']

    for topic_id in unique_topics:
        topic_dir_path = os.path.join(BASE_DIR, f'crf-tmp-{topic_id.strip()}')
        if not os.path.exists(topic_dir_path):
            print(f"Directory for topic {topic_id} does not exist.")
            continue
        sentence_metrics = train_and_predict_for_topic(topic_id, topic_dir_path)
        
        # Sentence Level Results
        print(f"Metrics for Topic {topic_id} at Sentence Level: Precision={sentence_metrics['Precision']}, Recall={sentence_metrics['Recall']}, F1={sentence_metrics['F1']}")

        # Annotation Level results
        metrics = evaluate_annotations(
            os.path.join(topic_dir_path, f"{topic_id}.gold.span"),
            os.path.join(topic_dir_path, f"{topic_id}.pred.span")
        )
        print(f"Metrics for Topic {topic_id} at Annotation Level: Precision={metrics['Precision']}, Recall={metrics['Recall']}, F1-Score={metrics['F1-Score']}")

if __name__ == '__main__':
    main()
