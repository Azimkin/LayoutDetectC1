from pathlib import Path
import os

from flask import Flask, jsonify, request

from model import LayoutDetectC1


def _model_path() -> Path:
    default_path = Path(__file__).resolve().parent / "out" / "LayoutDetectC1.pth"
    return Path(os.getenv("MODEL_PATH", default_path))


def _load_model() -> LayoutDetectC1:
    return LayoutDetectC1.load(str(_model_path()))


def _bad_request(message: str):
    return jsonify({"error": message}), 400


def create_app() -> Flask:
    app = Flask(__name__)
    model = _load_model()

    @app.post("/word")
    def word():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _bad_request("Expected a JSON object.")

        word = payload.get("word")
        if not isinstance(word, str) or not word.strip():
            return _bad_request("Field 'word' must be a non-empty string.")

        clazz = model.classify_word(word)
        return jsonify({
            "word": word,
            "class": clazz.name,
            "value": clazz.value,
        })

    @app.post("/phrase")
    def phrase():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _bad_request("Expected a JSON object.")

        phrase = payload.get("phrase")
        if not isinstance(phrase, str) or not phrase.strip():
            return _bad_request("Field 'phrase' must be a non-empty string.")

        clazz, values = model.classify_phrase(phrase)
        return jsonify({
            "phrase": phrase,
            "class": clazz.name,
            "value": clazz.value,
            "values": [
                {"class": value.name, "value": value.value}
                for value in values
            ],
        })

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
