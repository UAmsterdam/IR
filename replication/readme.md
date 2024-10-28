# CRF Model Training and Evaluation

## Overview

This project replicates the original paper’s implementation of Conditional Random Fields (CRF) for sequence learning tasks, translated into Python using CRFSuite. The process involves generating data folds from structured input data and training the CRF model to evaluate results based on precision, recall, and F1-score at both sentence and annotation levels.

## Requirements

### Software and Libraries
- Python 3.x
- Scikit-learn
- sklearn-crfsuite

### Installing Python Libraries

Install the required Python libraries using pip:

pip install scikit-learn
pip install sklearn-crfsuite 

### Data Preparation

Get the dataset from the original authors and keep the data in the same folder.

### Directory Structure
Ensure your data is organized under a main directory (data/) with the following subdirectories and files:

qrels/<topic_id>/<docid>.qrels: Label files for each document.
docs/<docid>.<featurization>: Document files containing feature representations.

### Generating Data Folds
Use the provided Bash script to generate data folds. The script processes input data into a format suitable for training with CRF.

### Usage
First, ensure the script generate_folds.sh is executable:

chmod +x generate_folds.sh

Run the script by providing the path to the data directory, the topic ID, and the featurization method:

./generate_folds.sh /path/to/data 1234 features

This command will create folds in the directory crf-tmp-<topic_id> within your main data directory.

### Running the Python Script
After generating the data folds, use the Python script to train and evaluate the CRF models. Ensure that you configure the BASE_DIR in the Python script to point to the location of your data folds.

### Execute the Script
Run the Python script from your command line:

python3 main.py

### Output
The script will output precision, recall, and F1-score for each topic at both sentence and span levels, printing the results directly to the console.

## Notes
Ensure all paths in the scripts are correctly set up according to your local environment.
The Bash script assumes a 5-fold configuration by default; adjust this as necessary for your dataset.
