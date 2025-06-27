from flask import Blueprint, render_template, request, jsonify
from app.core_ai.utils import get_response

public_bp = Blueprint("public", __name__)


@public_bp.route("/")
def landing_page():
    return render_template("public/landing.html")


@public_bp.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty input"}), 400

    response = get_response(user_input)
    print("response --> ", response)
    return jsonify({"answer": response})
