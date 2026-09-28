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
import warnings
import requests

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE = Path(__file__).resolve().parent
PUBLIC = BASE.parent
IMAGES = PUBLIC / 'images'
CACHE = BASE / 'data' / 'faces.json'
DOWNLOAD_CACHE = BASE / 'data' / 'download_cache'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
THRESHOLD = float(os.getenv('FACE_THRESHOLD', '0.42'))
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
    if IMAGES.exists() and any(IMAGES.iterdir()):
        for p in IMAGES.rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in EXTENSIONS:
                local_files[p.name] = p
        if local_files:
            return local_files

    # Remote sync from Hostinger
    DOWNLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(f"{REMOTE_GALLERY}/get_images.php", timeout=15, headers=headers)
        if r.status_code == 200:
            names = r.json()
            for name in names:
                clean_name = name.split('/')[-1]
                target_path = DOWNLOAD_CACHE / clean_name
                if not target_path.exists() or target_path.stat().st_size == 0:
                    try:
                        img_url = f"{REMOTE_GALLERY}/images/{requests.utils.quote(clean_name)}"
                        resp = requests.get(img_url, timeout=30, headers=headers)
                        if resp.status_code == 200:
                            target_path.write_bytes(resp.content)
                    except Exception as err:
                        logging.warning('Download error for %s: %s', clean_name, err)
                if target_path.exists() and target_path.stat().st_size > 0:
                    local_files[clean_name] = target_path
    except Exception as e:
        logging.warning('Remote gallery sync error: %s', e)

    return local_files


def decode(source):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(source) as picture:
                picture = ImageOps.exif_transpose(picture).convert('RGB')
                picture.thumbnail((1200, 1200))
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
        self.detector = cv2.FaceDetectorYN.create(str(detector), '', (320, 320), 0.70)
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
        except Exception as err:
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
    return jsonify(
        service='RANI AI Face Matching Engine',
        status='active',
        ready=state['ready'],
        total_photos=state['total'],
        face_photos=state['face_photos']
    )


@app.route('/api/status')
@app.route('/gaga_collection/api/status')
def status():
    with lock:
        return jsonify(dict(state))


@app.route('/api/sync')
def sync():
    refresh()
    return jsonify(dict(state))


@app.before_request
def ensure_indexed():
    if request.path.startswith('/api/'):
        if not state['ready'] or engine is None or len(records) == 0:
            if not state['indexing']:
                try:
                    refresh()
                except Exception as err:
                    logging.warning('Auto-indexing error: %s', err)


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

    if not state['ready'] or engine is None or len(records) == 0:
        try:
            refresh()
        except Exception as err:
            logging.error('On-demand refresh error: %s', err)

    with lock:
        if engine is None:
            return jsonify(error='AI engine is preparing. Please try again in a few seconds.'), 503
        features = engine.features(picture)
        if len(features) == 0:
            return jsonify(error='No face detected in the photo. Please use a clear, front-facing selfie.'), 400

        # Match across all faces detected in the uploaded photo
        query_vectors = [np.array(f, dtype=np.float32) for f in features]
        matches = []
        for name, entry in records.items():
            if not entry.get('faces'):
                continue
            face_matrix = np.array(entry['faces'], dtype=np.float32)
            best_score = 0.0
            for q in query_vectors:
                s = float(np.max(face_matrix @ q))
                if s > best_score:
                    best_score = s
            if best_score >= THRESHOLD:
                matches.append({'name': name, 'score': round(best_score, 4)})

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


start_indexer()
try:
    refresh()
except Exception as e:
    logging.warning('Initial indexing warning: %s', e)

if __name__ == '__main__':
    from waitress import serve
    port = int(os.getenv('PORT', '8093'))
    serve(app, host='0.0.0.0', port=port, threads=4)
