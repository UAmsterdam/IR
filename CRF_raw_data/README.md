
# CRF Model for Text Classification on RAW DATA

This folder contains code for training, evaluating, and testing a Conditional Random Field (CRF) model for text classification tasks. The experiment is structured to facilitate data loading, feature extraction, model training, and evaluation. In this experiment, we have used raw sentences instead of pre-computed features (shared by Adam).

## Code Structure

- **crf_model.py**: Contains functions for training and loading the CRF model.
- **data_loader.py**: Includes functions for loading and processing data.
- **evaluation.py**: Provides functions for evaluating model performance.
- **features.py**: Defines functions for feature extraction required for CRFSuite format.
- **main.py**: The main script for running the entire pipeline.
- **Notebook_version.ipynb**: Jupyter Notebook version of the main script for interactive use.
- **test.py**: Script for testing the trained model.
- **train.py**: Script for training the model.

## Requirements

To run the code, you need the following packages installed:

```bash
pip install pandas sklearn-crfsuite
```

## Usage

### Data Loading

The \`data_loader.py\` script provides functions to load and process data:

- \`load_data(file_path)\`: Loads data from a CSV file.
- \`read_split(file_path)\`: Reads and splits data from a file.
- \`map_doc(row)\`: Maps document ID from row.
- \`stratify(df)\`: Stratifies the dataframe for balanced splits.

### Feature Extraction

The \`features.py\` script contains functions to extract features:

- \`extract_features_and_labels(sentences, labels)\`: Extracts features and labels from sentences.
- \`process_text_and_extract_features(sentences)\`: Processes text and extracts features.

### Model Training

The \`crf_model.py\` script includes functions for training and loading the CRF model:

- \`train_crf_model(X_train, y_train, model_path)\`: Trains the CRF model and saves it to the specified path.
- \`load_crf_model(model_path)\`: Loads the CRF model from the specified path.

### Evaluation

The \`evaluation.py\` script provides functions for evaluating model performance:

- \`sentence_level_results(true_labels_flat, predicted_labels_flat)\`: Computes sentence-level evaluation metrics.
- \`load_labels_and_create_spans(labels)\`: Loads labels and creates spans for evaluation.
- \`evaluate_annotations(pred_annotations, gold_annotations)\`: Evaluates the annotations.
- \`evaluate_model_predictions(model, X_test, y_test)\`: Evaluates model predictions.

### Main Script

The \`main.py\` script orchestrates the entire pipeline from data loading to model evaluation:

Update all the paths according to your data structures like:
- path = '../../core-tech/core/qrels/'
- tokenizer_path = '../../core-tech/custom_punkt_tokenizer.pkl'
- data_path  = '../../core-tech/due_dilligence_data.csv'
- model_save_dir = 'raw_data_exp_25_04_24'

```bash
python main.py
```

### Training and Testing

To train the model, use the \`train.py\` script:

```bash
python train.py
```

To test the model, use the \`test.py\` script:

```bash
python test.py
```

### Jupyter Notebook

For an interactive version of the pipeline, use the \`Notebook_version.ipynb\` notebook. This is useful for debugging and exploratory data analysis.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

This project uses the \`sklearn-crfsuite\` package for implementing the CRF model. We appreciate the contributions of the open-source community in providing such valuable tools.
