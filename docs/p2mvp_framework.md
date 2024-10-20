# **Tênis Vídeo Treino (TVTx)**

## **Visão Geral do Projeto**
O *TVTx* é uma plataforma PaaS que utiliza visão computacional e aprendizado profundo para análise de desempenho em tênis. O sistema captura golpes de Forehand, Backhand e Saque em tempo real, processando os vídeos com **OpenCV** e analisando com **PyTorch**. O feedback é dado imediatamente através de uma interface simples e intuitiva criada com **Streamlit**.

### **Tecnologias Utilizadas**
- **Back-end:** Flask
- **Banco de Dados:** PostgreSQL
- **Visão Computacional:** OpenCV
- **Aprendizado de Máquina:** PyTorch, YOLO
- **Front-end:** Streamlit
- **Containers:** Docker

---

## **Estrutura de Arquivos**

Aqui está a estrutura de arquivos recomendada para o repositório GitHub:

```bash
tvtX/
│
├── README.md
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── app/
│   ├── __init__.py
│   ├── models.py
│   ├── routes.py
│   └── utils.py
│
├── config/
│   └── config.py
│
├── migrations/
│   └── versions/    # Arquivos de migração para o banco de dados
│
├── notebooks/
│   └── yolo_training.ipynb
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   └── base.html
│   └── index.html
│   └── analysis.html
│
├── tests/
│   ├── test_models.py
│   ├── test_routes.py
│   └── test_utils.py
│
└── requirements.txt
```

### **Principais diretórios:**
- `app/`: Contém a lógica do aplicativo Flask, incluindo modelos de banco de dados, rotas e utilitários.
- `config/`: Arquivos de configuração do projeto.
- `migrations/`: Scripts de migração do banco de dados gerados pelo **Flask-Migrate**.
- `notebooks/`: Notebooks para treinar ou ajustar modelos de aprendizado de máquina.
- `static/`: Arquivos estáticos como CSS e JavaScript.
- `templates/`: Arquivos HTML para as páginas da aplicação.
- `tests/`: Scripts de testes unitários e de integração.
- `requirements.txt`: Lista de pacotes Python necessários.

---

## **Configuração do Ambiente de Desenvolvimento**

### **1. Clonar o repositório**

```bash
git clone https://github.com/usuario/tvtx.git
cd tvtx
```

### **2. Configuração do ambiente virtual**

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows
```

### **3. Instalar dependências**

```bash
pip install -r requirements.txt
```

### **4. Configurar as variáveis de ambiente**
Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```bash
FLASK_APP=run.py
FLASK_ENV=development
DATABASE_URL=postgresql://usuario:senha@localhost/tvtx_db
```

### **5. Rodar as migrações do banco de dados**

```bash
flask db init
flask db migrate
flask db upgrade
```

### **6. Subir a aplicação com Docker**

Certifique-se de que você tem Docker instalado e configurado.

```bash
docker-compose up --build
```

### **7. Acessar a aplicação**

Acesse `http://localhost:5000` no navegador para verificar se a aplicação está rodando.

---

## **Exemplo de Código**

### **1. `app/__init__.py`** – Inicialização do Flask e Banco de Dados

```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.config.DevelopmentConfig')
    
    db.init_app(app)
    migrate = Migrate(app, db)

    from .routes import main
    app.register_blueprint(main)

    return app
```

### **2. `app/models.py`** – Modelo de Usuário e Registro de Análise

```python
from . import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_path = db.Column(db.String(200), nullable=False)
    feedback = db.Column(db.Text, nullable=False)
```

### **3. `app/routes.py`** – Rotas do Flask

```python
from flask import Blueprint, render_template, request, redirect, url_for
from .models import User, Analysis
from .utils import analyze_video
from . import db

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analysis', methods=['POST'])
def analyze():
    video = request.files['video']
    feedback = analyze_video(video)
    analysis = Analysis(user_id=current_user.id, video_path=video.filename, feedback=feedback)
    db.session.add(analysis)
    db.session.commit()
    return redirect(url_for('main.index'))
```

### **4. `app/utils.py`** – Análise com OpenCV e PyTorch

```python
import cv2
import torch
from yolo import YOLOv5

def analyze_video(video):
    # Carregar o vídeo usando OpenCV
    cap = cv2.VideoCapture(video)
    model = YOLOv5("best.pt")  # Modelo YOLO pré-treinado
    
    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Realizar a inferência no frame
        detections = model(frame)
        results.append(detections)
    
    cap.release()
    return generate_feedback(results)

def generate_feedback(results):
    feedback = "Resumo da análise: \n"
    for result in results:
        feedback += f"{result['label']}: {result['confidence'] * 100:.2f}% de confiança.\n"
    return feedback
```

---

## **Testes Unitários**

### **1. `tests/test_models.py`**

```python
import unittest
from app import create_app, db
from app.models import User

class UserModelTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_user_creation(self):
        user = User(username='testuser', email='test@test.com', password='password')
        db.session.add(user)
        db.session.commit()
        self.assertTrue(User.query.filter_by(username='testuser').first() is not None)
```

---

## **Conclusão**

O *TVTx* é um projeto que combina o poder de visão computacional com aprendizado de máquina para oferecer uma análise detalhada e em tempo real do desempenho em tênis. Utilizando uma stack moderna com Flask, OpenCV, PyTorch e Streamlit, o sistema é capaz de capturar vídeos, processá-los e fornecer feedback imediato aos usuários, permitindo um avanço significativo na forma como jogadores de tênis monitoram e melhoram suas habilidades.
