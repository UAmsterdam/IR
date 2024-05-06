from data_loader import load_data, read_split, stratify, map_doc
from crf_model import train_crf_model, load_crf_model, predict_with_crf
from tokenizer import load_tokenizer
from features import extract_features_and_labels, process_text_and_extract_features
from evaluation import sentence_level_results, load_labels_and_create_spans, evaluate_annotations, evaluate_model_predictions
from train import train_loop
from test import test_loop
import pandas as pd

import os

path = '../../core-tech/core/qrels/'
tokenizer_path = '../../core-tech/custom_punkt_tokenizer.pkl'
data_path  = '../../core-tech/due_dilligence_data.csv'
model_save_dir = 'raw_data_exp_25_04_24'

# List of folder/topic names
# folders_to_process = ['1272', '1474','1238', '1275', '1239', '1520', '1509', '1240', '1308', '1319', '1439', 
#                  '1267', '1242', '1462', '1265', '1444', '1312', '1244', '1243', '1468', '1309', '1524', 
#                  '1247', '1440', '1251', '1249', '1248', '1262', '1250', '1252', '1245', '1512', '1498', 
#                  '1601', '1443', '1086', '1551', '1253', '1320', '1304', '1469', '1611', '1300', '1489', 
#                  '1500', '1261', '1318', '1460', '1475', '1321']

folders_to_process = [ '1524']

## Load the custom tokenizer
tokenizer = load_tokenizer(tokenizer_path)

## Load the data
df = load_data(data_path)

# Iterate over folders
for folder in folders_to_process:
    
    ## Dataframe to store the results
    results_df = pd.DataFrame(columns=["Topic ID", "Fold", "Sentence Precision", "Sentence Recall", "Sentence F1-Score",
                               "Annotation TP", "Annotation FP", "Annotation FN", 
                               "Annotation Precision", "Annotation Recall", "Annotation F1-Score"])


    if folder in os.listdir(path):
        for fold in range(5):
            
            ####### Data split into train and test sets #######
            print("For fold:", fold)
            test_split = fold
            train_split = [i for i in range(5) if i != test_split]
            
            test = read_split(f'{path}/{folder}/{folder}-{test_split}.cache')
            train = sum([read_split(f'{path}/{folder}/{folder}-{el}.cache') for el in train_split], [])

            df_ = df[df['topic_id'] == int(folder)].dropna()
#             df_ = df_.dropna()
            df_['doc_id'] = df_.apply(map_doc, axis=1)

            df_train = df_[df_['doc_id'].isin(train)]
            df_test = df_[df_['doc_id'].isin(test)]
                    
            ## genertae test data
            test_unbalanced, test_balanced = stratify(df_test)
            train_unbalanced, train_balanced = stratify(df_train)
            
            ######### Model training ##############

            model_save_dir = 'raw_data_exp_25_04_24'
            os.makedirs(model_save_dir, exist_ok=True)
            
            ## train CRF model
            train_loop(train_unbalanced, model_save_dir, str(int(folder)), fold, tokenizer)
            
            ## test CRF model
            results_df = test_loop(test_unbalanced, model_save_dir, folder, fold, results_df, tokenizer)
    
    
    print("\n Completed all folds")
    # Save results to CSV
    results_csv_path = os.path.join(model_save_dir, folder, "final_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")


### Calculating mean across topics -- optional 

dataframes = []
# Step 1: Loop through each topic ID and read the CSV file
for topic_id in folders_to_process:
    csv_path = os.path.join(model_save_dir, str(topic_id), "final_results.csv")
    #print(csv_path)
    df = pd.read_csv(csv_path)
    dataframes.append(df)

# Step 2: Concatenate all DataFrames
combined_df = pd.concat(dataframes)

# Calculate mean for each topic across all folds
topic_means = combined_df.groupby('Topic ID').mean().drop(columns='Fold')

# Calculate the mean of means across all topics
overall_means = topic_means.mean()

# Print results
print("Mean Metrics for Each Topic:")
print(topic_means)
print("\nOverall Mean Metrics Across All Topics:")
print(overall_means)