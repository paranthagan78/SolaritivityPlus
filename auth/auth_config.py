"""auth/auth_config.py"""
import hashlib, hmac, time, os
from functools import wraps
from flask import request, jsonify, session
from config import PASSCODE, SESSION_TIMEOUT, MAX_ATTEMPTS, LOCKOUT_DURATION

_SALT = os.urandom(32)
_PASSCODE_HASH = hashlib.pbkdf2_hmac("sha256", PASSCODE.encode(), _SALT, iterations=260_000)
_attempt_tracker: dict = {}

def _hash_candidate(c): return hashlib.pbkdf2_hmac("sha256", c.encode(), _SALT, iterations=260_000)
def _get_ip(): return request.headers.get("X-Forwarded-For", request.remote_addr)

def is_locked_out(ip):
    r = _attempt_tracker.get(ip)
    if not r: return False, 0
    if time.time() < r.get("locked_until", 0):
        return True, int(r["locked_until"] - time.time())
    return False, 0

def record_failed(ip):
    r = _attempt_tracker.setdefault(ip, {"count": 0, "locked_until": 0})
    r["count"] += 1
    if r["count"] >= MAX_ATTEMPTS:
        r["locked_until"] = time.time() + LOCKOUT_DURATION
        r["count"] = 0
    return r["count"]

def clear_attempts(ip): _attempt_tracker.pop(ip, None)
def validate_passcode(c): return hmac.compare_digest(_hash_candidate(c), _PASSCODE_HASH)

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated") or (time.time() - session.get("auth_time", 0)) > SESSION_TIMEOUT:
            session.clear()
            return jsonify({"success": False, "error": "Unauthorized", "code": "AUTH_REQUIRED"}), 401
        return f(*args, **kwargs)
    return decorated