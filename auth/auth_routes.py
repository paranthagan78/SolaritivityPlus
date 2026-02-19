"""auth/auth_routes.py"""
import time
from flask import Blueprint, request, jsonify, session
from .auth_config import validate_passcode, is_locked_out, record_failed, clear_attempts, _get_ip, MAX_ATTEMPTS, LOCKOUT_DURATION, SESSION_TIMEOUT

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

@auth_bp.route("/login", methods=["POST"])
def login():
    ip = _get_ip()
    locked, remaining = is_locked_out(ip)
    if locked:
        return jsonify({"success": False, "error": f"Locked out. Try in {remaining}s.", "code": "LOCKED_OUT", "retry_after": remaining}), 429
    data = request.get_json(silent=True) or {}
    passcode = data.get("passcode", "").strip()
    if not passcode:
        return jsonify({"success": False, "error": "Passcode required."}), 400
    if validate_passcode(passcode):
        clear_attempts(ip)
        session.clear()
        session["authenticated"] = True
        session["auth_time"] = time.time()
        session.permanent = True
        return jsonify({"success": True, "message": "Authenticated.", "session_timeout": SESSION_TIMEOUT}), 200
    count = record_failed(ip)
    locked_now, lock_rem = is_locked_out(ip)
    if locked_now:
        return jsonify({"success": False, "error": f"Locked out for {LOCKOUT_DURATION}s.", "code": "LOCKED_OUT", "retry_after": lock_rem}), 429
    return jsonify({"success": False, "error": "Invalid passcode.", "code": "INVALID_PASSCODE", "attempts_remaining": max(0, MAX_ATTEMPTS - count)}), 401

@auth_bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out."}), 200

@auth_bp.route("/status", methods=["GET"])
def status():
    if session.get("authenticated") and (time.time() - session.get("auth_time", 0)) <= SESSION_TIMEOUT:
        return jsonify({"authenticated": True, "session_remaining": int(SESSION_TIMEOUT - (time.time() - session["auth_time"]))}), 200
    session.clear()
    return jsonify({"authenticated": False}), 200