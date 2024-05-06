import os
import pickle
import sklearn_crfsuite

def train_crf_model(X_train, y_train, model_path):
    crf = sklearn_crfsuite.CRF(algorithm='pa', c=0.1, pa_type=2, max_iterations=100,
                               all_possible_transitions=True, verbose=True)
    crf.fit(X_train, y_train)

    with open(model_path, "wb") as model_file:
        pickle.dump(crf, model_file)

def load_crf_model(model_path):
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)
    
def predict_with_crf(crf_model, X_test):
    """Predict using a loaded CRF model."""
    return crf_model.predict(X_test)
