"""Database-free folder album with YuNet detection and SFace matching.
Supports standalone Cloud deployment (Render) with remote Hostinger gallery sync.
"""
import io
import json
import logging
import os
from pathlib import Path
import threading
import time
import urllib.request
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

BASE = Path(__file__).resolve().parent
PUBLIC = BASE.parent
IMAGES = PUBLIC / 'images'
CACHE = BASE / 'data' / 'faces.json'
DOWNLOAD_CACHE = BASE / 'data' / 'download_cache'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
THRESHOLD = float(os.getenv('FACE_THRESHOLD', '0.45'))
REMOTE_GALLERY = os.getenv('REMOTE_GALLERY', 'https://rarebook.in/gaga_collection').rstrip('/')
Image.MAX_IMAGE_PIXELS = 40_000_000

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
app.config['MAX_CONTENT_LENGTH'] = 35 * 1024 * 1024

lock = threading.RLock()
engine = None
state = {'ready': False, 'indexing': False, 'total': 0, 'processed': 0,
         'face_photos': 0, 'skipped': 0, 'error': None}
records = {}

# Ensure models are downloaded on startup
from setup_models import setup as download_models_if_needed
try:
    download_models_if_needed()
except Exception as e:
    logging.warning('Model setup error: %s', e)


def safe_files():
    local_files = {}
    if IMAGES.exists():
        local_files = {p.relative_to(IMAGES).as_posix(): p for p in IMAGES.rglob('*')
                       if p.is_file() and not p.is_symlink() and p.suffix.lower() in EXTENSIONS}
    if local_files:
        return local_files

    # Remote fallback to Hostinger
    DOWNLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(f"{REMOTE_GALLERY}/get_images.php", headers={'User-Agent': 'RaniAI/1.0'})
        with urllib.request.urlopen(req, timeout=15) as res:
            img_list = json.loads(res.read().decode())
            for name in img_list:
                local_path = DOWNLOAD_CACHE / name
                if not local_path.exists():
                    try:
                        img_url = f"{REMOTE_GALLERY}/images/{urllib.parse.quote(name)}"
                        urllib.request.urlretrieve(img_url, local_path)
                    except Exception as err:
                        logging.warning('Could not download %s: %s', name, err)
                if local_path.exists():
                    local_files[name] = local_path
    except Exception as e:
        logging.warning('Remote gallery sync error: %s', e)

    return local_files


def decode(source):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(source) as picture:
                picture = ImageOps.exif_transpose(picture).convert('RGB')
                picture.thumbnail((2000, 2000))
                return cv2.cvtColor(np.array(picture), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('Please use a valid JPG, PNG or WebP photo under 40 megapixels.') from exc


class Faces:
    def __init__(self):
        detector = BASE / 'models' / 'face_detection_yunet_2023mar.onnx'
        recognizer = BASE / 'models' / 'face_recognition_sface_2021dec.onnx'
        if not detector.exists() or not recognizer.exists():
            download_models_if_needed()
        self.detector = cv2.FaceDetectorYN.create(str(detector), '', (320, 320), 0.85)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer), '')

    def features(self, picture):
        self.detector.setInputSize((picture.shape[1], picture.shape[0]))
        _, faces = self.detector.detect(picture)
        result = []
        for face in ([] if faces is None else faces):
            aligned = self.recognizer.alignCrop(picture, face)
            feature = self.recognizer.feature(aligned).flatten()
            feature /= max(float(np.linalg.norm(feature)), 1e-12)
            result.append(feature.tolist())
        return result


def refresh():
    global records, engine
    files = safe_files()
    with lock:
        state.update(indexing=True, total=len(files), processed=0, skipped=0, error=None)
        if engine is None:
            engine = Faces()
    updated = {}
    for name, path in sorted(files.items()):
        try:
            stat = path.stat()
            stamp = [stat.st_mtime_ns, stat.st_size]
            with lock:
                previous = records.get(name)
                if previous and previous.get('stamp') == stamp:
                    entry = previous
                else:
                    entry = {'stamp': stamp, 'faces': engine.features(decode(path))}
            updated[name] = entry
        except (ValueError, OSError, cv2.error) as err:
            logging.warning('Skipped unreadable photo %s: %s', name, err)
            with lock:
                state['skipped'] += 1
        finally:
            with lock:
                state['processed'] += 1
    CACHE.parent.mkdir(exist_ok=True)
    temporary = CACHE.with_suffix('.tmp')
    temporary.write_text(json.dumps({'version': 1, 'photos': updated}))
    temporary.replace(CACHE)
    with lock:
        records = updated
        state.update(ready=True, indexing=False,
                     face_photos=sum(bool(item['faces']) for item in records.values()))


def index_loop():
    global records
    try:
        if CACHE.exists():
            saved = json.loads(CACHE.read_text())
            if saved.get('version') == 1:
                records = saved['photos']
                state.update(ready=True, total=len(records),
                             face_photos=sum(bool(item['faces']) for item in records.values()))
    except (OSError, ValueError, KeyError):
        records = {}
    while True:
        try:
            refresh()
        except Exception as e:
            logging.exception('Album indexing failed: %s', e)
            with lock:
                state.update(indexing=False, error='Album indexing failed. Check server logs.')
        time.sleep(30)


@app.route('/')
def home():
    return jsonify(service='RANI AI Face Matching Engine', status='active', ready=state['ready'], total_photos=state['total'])


@app.route('/api/status')
@app.route('/gaga_collection/api/status')
def status():
    with lock:
        return jsonify(dict(state))


@app.route('/api/search', methods=['POST'])
@app.route('/gaga_collection/api/search', methods=['POST'])
def search():
    uploaded = request.files.get('photo')
    if not uploaded:
        return jsonify(error='Please choose a selfie photo first.'), 400
    try:
        picture = decode(io.BytesIO(uploaded.read()))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    with lock:
        if not state['ready'] or state['error']:
            if engine is None:
                return jsonify(error='AI engine is preparing. Please try again in 5 seconds.'), 503
        features = engine.features(picture)
        if len(features) == 0:
            return jsonify(error='No face detected in the photo. Please use a clear, front-facing selfie.'), 400
        if len(features) > 1:
            return jsonify(error='Multiple faces detected. Please upload a photo with only your face.'), 400

        query = np.array(features[0], dtype=np.float32)
        matches = []
        for name, entry in records.items():
            if not entry.get('faces'):
                continue
            face_matrix = np.array(entry['faces'], dtype=np.float32)
            score = float(np.max(face_matrix @ query))
            if score >= THRESHOLD:
                matches.append({'name': name, 'score': round(score, 4)})

        matches.sort(key=lambda item: item['score'], reverse=True)

    return jsonify(matches=matches, count=len(matches))


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-CSRF-Token, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    if '/api/' in request.path:
        response.headers['Cache-Control'] = 'no-store'
    return response


def start_indexer():
    threading.Thread(target=index_loop, daemon=True, name='album-indexer').start()


# Auto-start indexing on import (for gunicorn/waitress)
start_indexer()

if __name__ == '__main__':
    from waitress import serve
    port = int(os.getenv('PORT', '8093'))
    serve(app, host='0.0.0.0', port=port, threads=4)
