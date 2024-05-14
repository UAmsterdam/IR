
# Bi-LSTM Model for Text Analysis with Attention Layer

This Experiment contains code for training, evaluating, and testing a deep learning model for text analysis tasks using LSTM networks and an attention mechanism. The project is structured to provide data loading, model training, and evaluation using TensorFlow and Keras.

## Folder Structure

- **attention.py**: Implements the attention mechanism for the LSTM model.
- **callbacks.py**: Includes a custom callback for calculating metrics during training.
- **evaluation.py**: Provides functions for evaluating model performance.
- **main.py**: The main script for running the entire pipeline.
- **Notebook_version.ipynb**: Jupyter Notebook version of the main script for interactive use.
- **train.py**: Script for training the model.

## Requirements

To run the code, you need the following packages installed:

\`\`\`bash
pip install tensorflow numpy pandas scikit-learn nltk
\`\`\`

### Data Loading

Data is loaded and preprocessed through the main script, which integrates the data loading functionalities necessary for preparing the input for the model.

### Model Training

The \`train.py\` script includes functions for setting up and training the model:

- \`train_evaluate_model(data, labels)\`: Handles the training and evaluation of the model. Change the training parametrrs according to your requirements.

### Evaluation

The \`evaluation.py\` script provides functions for detailed evaluation of the model performance:

- \`evaluate_model(model, X_test, y_test)\`: Evaluates the model on the test set.

### Main Script

The \`main.py\` script orchestrates the entire pipeline from data loading to model evaluation:

\`\`\`bash
python main.py
\`\`\`

### Jupyter Notebook

For an interactive version of the pipeline, use the \`Notebook_version.ipynb\` notebook. This is useful for debugging, exploratory data analysis, and visualizing the performance of the model. In the Notebook, we have demonstrated an example training for 1 epoch. 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

This project utilizes TensorFlow and the Keras API to implement the deep learning model. We appreciate the contributions of the open-source community in providing these powerful tools for machine learning research and application.
