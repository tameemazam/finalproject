import joblib


model_path = ('student_classifier.joblib')
def load_trained_model(model_path):
    # Loads a model from the specified path and returns it
    return joblib.load(model_path)
