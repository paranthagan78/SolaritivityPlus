"""modules/el_upload/upload_utils.py"""
import os, uuid
from werkzeug.utils import secure_filename
from config import ALLOWED_EXTENSIONS, UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_upload(file):
    """Save file with a unique name; return (saved_filename, full_path)."""
    original = secure_filename(file.filename)
    ext = original.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(path)
    return unique_name, path, original

def get_file_info(filename, original_name, path):
    size = os.path.getsize(path)
    return {
        "filename": filename,
        "original_name": original_name,
        "size_kb": round(size / 1024, 2),
        "url": f"/uploads/{filename}",
    }