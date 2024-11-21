from flask import Blueprint, request, jsonify
from .models import StrokeRecord
from .database import db
from .ml_model import predict_stroke, retrain_model

api_blueprint = Blueprint("api", __name__)

@api_blueprint.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data.get("features")
    handedness = data.get("handedness")
    stroke_type = predict_stroke(features, handedness)
    return jsonify({"stroke_type": stroke_type})

@api_blueprint.route("/record", methods=["POST"])
def record():
    data = request.json
    new_record = StrokeRecord(
        stroke_type=data["stroke_type"],
        handedness=data["handedness"],
        features=data["features"]
    )
    db.session.add(new_record)
    db.session.commit()
    return jsonify({"message": "Record saved successfully!"})

@api_blueprint.route("/retrain", methods=["POST"])
def retrain():
    records = StrokeRecord.query.all()
    features = [r.features for r in records]
    labels = [r.stroke_type for r in records]
    retrain_model(features, labels)
    return jsonify({"message": "Model retrained successfully!"})
