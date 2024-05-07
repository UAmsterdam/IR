import numpy as np
import os
import pickle
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score, f1_score

class MetricsAfterEpoch(Callback):
    """
    Custom callback to calculate precision, recall, and F1-score after each specified interval of epochs.
    """
    def __init__(self, X_train, y_train, X_test, y_test, interval=5):
        super(MetricsAfterEpoch, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            y_pred = self.model.predict(self.X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            precision = precision_score(self.y_test, y_pred_binary)
            recall = recall_score(self.y_test, y_pred_binary)
            f1 = f1_score(self.y_test, y_pred_binary)
            print(f"\nAfter epoch {epoch+1}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")

class SaveModelAndTokenizerCallback(Callback):
    """
    Custom callback to save the model and tokenizer at specified intervals.
    """
    def __init__(self, topic_id, model_save_path, tokenizer, tokenizer_save_path, interval):
        super(SaveModelAndTokenizerCallback, self).__init__()
        self.topic_id = topic_id
        self.model_save_path = model_save_path
        self.tokenizer = tokenizer
        self.tokenizer_save_path = tokenizer_save_path
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            model_dir = os.path.join(self.model_save_path, str(self.topic_id))
            tokenizer_dir = os.path.join(self.tokenizer_save_path, str(self.topic_id))
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(tokenizer_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{self.topic_id}_model_at_epoch_{epoch + 1}.h5")
            self.model.save(model_path)
            tokenizer_path = os.path.join(tokenizer_dir, f"{self.topic_id}_tokenizer_at_epoch_{epoch + 1}.pickle")
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Model and tokenizer saved to {model_path} and {tokenizer_path}")

class SavePredAndGoldCallback(Callback):
    """
    Custom callback to save predictions and gold labels at specified intervals.
    """
    def __init__(self, X_test, y_test, model_save_dir, topic_id, interval=5):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.model_save_dir = model_save_dir
        self.interval = interval
        self.topic_id = topic_id

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            topic_dir = os.path.join(self.model_save_dir, str(self.topic_id))
            os.makedirs(topic_dir, exist_ok=True)
            y_pred = self.model.predict(self.X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            y_pred_str = [str(pred[0]) for pred in y_pred_binary]
            y_test_str = [str(label) for label in self.y_test]
            pred_file_path = os.path.join(topic_dir, f"{self.topic_id}_epoch_{epoch+1}.pred.raw")
            gold_file_path = os.path.join(topic_dir, f"{self.topic_id}_epoch_{epoch+1}.gold.raw")
            with open(pred_file_path, "w") as pred_file, open(gold_file_path, "w") as gold_file:
                pred_file.write("\n".join(y_pred_str))
                gold_file.write("\n".join(y_test_str))
            print(f"Predictions and gold labels saved for epoch {epoch+1} in topic {self.topic_id}")
