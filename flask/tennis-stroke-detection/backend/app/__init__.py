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
