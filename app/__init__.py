from flask import Flask


def create_app():
    app = Flask(__name__)

    from app.public.views import public_bp

    app.register_blueprint(public_bp)

    return app
