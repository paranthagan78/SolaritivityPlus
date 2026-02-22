"""
app.py — Solar PV Defect Detection System
Run with: python app.py
"""
import os
import secrets
from flask import Flask, jsonify, send_from_directory, render_template
from flask_cors import CORS


def create_app() -> Flask:
    app = Flask(__name__)

    # ── Import config INSIDE factory to avoid circular import ─────────────
    import config as cfg

    # ── Security ──────────────────────────────────────────────────────────
    app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = (
        os.environ.get("PRODUCTION", "false").lower() == "true"
    )
    app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_CONTENT_LENGTH

    # ── CORS ──────────────────────────────────────────────────────────────
    CORS(app, supports_credentials=True,
         origins=[
             "http://localhost:3000",
             "http://127.0.0.1:5000",
             "http://0.0.0.0:5000",
         ])

    # ── Register blueprints ────────────────────────────────────────────────
    from auth import auth_bp
    from modules.el_upload   import el_bp
    from modules.integrity   import integrity_bp
    from modules.detection   import detection_bp
    from modules.thermal     import thermal_bp
    from modules.carbon      import carbon_bp
    from modules.xai         import xai_bp
    from modules.llm_summary import llm_bp
    from modules.chatbot     import chatbot_bp
    from modules.weather     import weather_bp

    for bp in (auth_bp, el_bp, integrity_bp, detection_bp,
               thermal_bp, carbon_bp, xai_bp, llm_bp,
               chatbot_bp, weather_bp):
        app.register_blueprint(bp)

    # ── Eagerly load DistilBERT at startup ────────────────────────────────
    # THIS IS THE KEY FIX for the teammate problem.
    #
    # Without this, the model loads lazily on the first request. On a machine
    # that hasn't downloaded the model yet (~67 MB), the download happens during
    # that first request — and every request that arrives before the download
    # completes silently falls back to rule-based sentiment instead of DistilBERT.
    #
    # With this call, the model is fully downloaded, loaded, and warmed up
    # BEFORE Flask starts serving requests. Every machine gets DistilBERT
    # from request #1, regardless of whether the model was pre-cached.
    from modules.chatbot.sentiment import warmup_sentiment_model
    warmup_sentiment_model()

    # ── Static file routes ─────────────────────────────────────────────────
    @app.route("/uploads/<path:filename>")
    def serve_upload(filename):
        return send_from_directory(cfg.UPLOAD_FOLDER, filename)

    @app.route("/results/<path:filename>")
    def serve_result(filename):
        return send_from_directory(cfg.RESULT_FOLDER, filename)

    @app.route("/explanations/<path:filename>")
    def serve_explanation(filename):
        return send_from_directory(cfg.EXPLAIN_FOLDER, filename)

    @app.route("/thermal_uploads/<path:filename>")
    def serve_thermal(filename):
        return send_from_directory(cfg.THERMAL_UPLOAD_FOLDER, filename)

    # ── Page routes ────────────────────────────────────────────────────────
    @app.route("/")
    @app.route("/login")
    def login_page():
        return render_template("login.html")

    @app.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    # ── Health check ───────────────────────────────────────────────────────
    # Now includes sentiment model status so you can verify DistilBERT loaded
    # correctly on any machine by hitting GET /api/health
    @app.route("/api/health")
    def health():
        from modules.chatbot.sentiment import _model_loaded, _model_failed
        if _model_loaded:
            sentiment_status = "distilbert_ready"
        elif _model_failed:
            sentiment_status = "rule_based_fallback"
        else:
            sentiment_status = "not_loaded"

        return jsonify({
            "status":          "ok",
            "service":         "Solar PV Defect Detection",
            "sentiment_model": sentiment_status,
        }), 200

    # ── Error handlers ─────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Endpoint not found."}), 404

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": "File too large. Maximum is 50 MB."}), 413

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error.", "detail": str(e)}), 500

    return app


if __name__ == "__main__":
    application = create_app()
    print("\n" + "=" * 55)
    print("  Solar PV Defect Detection System")
    print("  http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    application.run(debug=True, host="0.0.0.0", port=5000)