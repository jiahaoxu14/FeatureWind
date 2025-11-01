from flask import Flask
from flask_cors import CORS
import os


def create_app():
    app = Flask(__name__)

    # Enable CORS for local dev (React on a different port)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register API blueprint (support running as script or module)
    try:
        from .api.routes import api_bp  # when run as module: python -m server.app
    except Exception:
        # when run as script: python server/app.py
        import os, sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from server.api.routes import api_bp
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
