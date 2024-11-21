import pickle
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "ml_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = RandomForestClassifier()
    model.fit([[0]*6], ["Unknown"])  # Dummy initial training

def predict_stroke(features, handedness):
    if hasattr(model, "predict"):
        return model.predict([features])[0]
    return "Unknown"

def retrain_model(features, labels):
    global model
    model.fit(features, labels)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
