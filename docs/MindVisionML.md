Here is a `.sh` script to automate the creation of the engineered solution, including generating the folder structure, placing all files, and deploying the solution using Docker Compose.

---

### **Script: `deploy_tennis_app.sh`**

```bash
#!/bin/bash

# Set up directories
echo "Creating folder structure..."
mkdir -p tennis-stroke-detection/{backend/app,frontend}
cd tennis-stroke-detection || exit

# Backend files
echo "Setting up backend files..."
cat > backend/app/__init__.py << 'EOF'
from flask import Flask
from .database import init_db

def create_app():
    app = Flask(__name__)
    app.config.from_mapping(
        SQLALCHEMY_DATABASE_URI="postgresql://user:password@db:5432/tennis",
        SQLALCHEMY_TRACK_MODIFICATIONS=False
    )
    init_db(app)
    from .routes import api_blueprint
    app.register_blueprint(api_blueprint)
    return app
EOF

cat > backend/app/models.py << 'EOF'
from .database import db

class StrokeRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stroke_type = db.Column(db.String(50), nullable=False)
    handedness = db.Column(db.String(10), nullable=False)
    features = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
EOF

cat > backend/app/routes.py << 'EOF'
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
EOF

cat > backend/app/database.py << 'EOF'
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
EOF

cat > backend/app/ml_model.py << 'EOF'
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
EOF

cat > backend/Dockerfile << 'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY ./app /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["flask", "run", "--host=0.0.0.0"]
EOF

cat > backend/requirements.txt << 'EOF'
Flask
Flask-SQLAlchemy
psycopg2-binary
scikit-learn
EOF

# Frontend files
echo "Setting up frontend files..."
cat > frontend/app.py << 'EOF'
import streamlit as st
import requests
import json

API_URL = "http://backend:5000"

st.title("Tennis Stroke Detection")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file:
    st.text("Processing video...")
    features = [0.5, 0.2, -0.1, 0.8, -0.2, 0.4]  # Dummy features
    handedness = st.selectbox("Select handedness", ["Right", "Left"])

    response = requests.post(f"{API_URL}/predict", json={"features": features, "handedness": handedness})
    if response.status_code == 200:
        stroke_type = response.json()["stroke_type"]
        st.write(f"Predicted Stroke Type: {stroke_type}")

        if st.button("Save Stroke"):
            requests.post(f"{API_URL}/record", json={"features": features, "handedness": handedness, "stroke_type": stroke_type})
            st.write("Stroke data saved!")
EOF

cat > frontend/Dockerfile << 'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

cat > frontend/requirements.txt << 'EOF'
streamlit
requests
EOF

# Docker Compose file
echo "Setting up Docker Compose..."
cat > docker-compose.yml << 'EOF'
version: '3.9'
services:
  backend:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - FLASK_APP=__init__.py
    depends_on:
      - db

  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    depends_on:
      - backend

  db:
    image: postgres:13
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: tennis
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
EOF

# Environment file
cat > .env << 'EOF'
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=tennis
EOF

# Build and start services
echo "Building and starting Docker containers..."
docker-compose up --build -d

echo "Deployment complete!"
echo "Frontend: http://localhost:8501"
echo "Backend: http://localhost:5000"
```

---

### **How to Use the Script**

1. **Save the script**: Save the script as `deploy_tennis_app.sh`.
2. **Make it executable**:
   ```bash
   chmod +x deploy_tennis_app.sh
   ```
3. **Run the script**:
   ```bash
   ./deploy_tennis_app.sh
   ```

The script will create the project structure, generate all required files, build the Docker images, and start the services. The frontend will be available at `http://localhost:8501` and the backend at `http://localhost:5000`.