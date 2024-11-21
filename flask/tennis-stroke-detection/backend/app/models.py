from .database import db

class StrokeRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stroke_type = db.Column(db.String(50), nullable=False)
    handedness = db.Column(db.String(10), nullable=False)
    features = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
