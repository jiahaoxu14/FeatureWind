from flask import Flask
from flask_cors import CORS
import os
import sys
from pathlib import Path


def create_app():
    # Ensure the core package is importable when running directly from backend/
    backend_root = Path(__file__).resolve().parent
    src_root = backend_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    app = Flask(__name__)

    # Enable CORS for local dev (React on a different port)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register API blueprint
    from routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


# Flask default entrypoint
app = create_app()

if __name__ == "__main__":
    # For development use only
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=True)
