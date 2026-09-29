"""Database-free folder album with YuNet detection and SFace matching.
Supports standalone Cloud deployment (Render) with remote Hostinger gallery sync.
"""
import gc
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

BASE = Path(__file__).resolve().parent
PUBLIC = BASE.parent
IMAGES = PUBLIC / 'images'
CACHE = BASE / 'data' / 'faces.json'
DOWNLOAD_CACHE = BASE / 'data' / 'index_work'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
INDEX_VERSION = 3
THRESHOLD = float(os.getenv('FACE_THRESHOLD', '0.42'))
BUILD = 'gaga-index-v3'
GALLERY_SOURCE = os.getenv('GALLERY_SOURCE', 'remote').lower()
cv2.setNumThreads(1)
REMOTE_GALLERY = os.getenv('REMOTE_GALLERY', 'https://rarebook.in/gaga_collection').rstrip('/')
Image.MAX_IMAGE_PIXELS = 40_000_000

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
app.config['MAX_CONTENT_LENGTH'] = 35 * 1024 * 1024

lock = threading.RLock()
engine_lock = threading.Lock()
refresh_lock = threading.Lock()
engine = None
state = {'ready': False, 'indexing': False, 'total': 0, 'processed': 0,
         'face_photos': 0, 'skipped': 0, 'error': None,
         'build': BUILD, 'source': GALLERY_SOURCE, 'stage': 'starting',
         'current_photo': None, 'last_photo_error': None}
records = {}

from setup_models import setup as download_models_if_needed


def gallery_files():
    if GALLERY_SOURCE == 'local':
        return {p.name: p for p in IMAGES.glob('*')
                if p.is_file() and not p.is_symlink() and p.suffix.lower() in EXTENSIONS}
    response = requests.get(f'{REMOTE_GALLERY}/get_images.php', timeout=(10, 30),
                            headers={'User-Agent': 'GAGA-Indexer/3'})
    response.raise_for_status()
    names = response.json()
    if not isinstance(names, list) or not names:
        raise ValueError('Gallery returned an empty or invalid photo list')
    for name in names:
        if (not isinstance(name, str) or Path(name).name != name or
                '\\' in name or Path(name).suffix.lower() not in EXTENSIONS):
            raise ValueError('Gallery returned an invalid filename')
    return dict.fromkeys(names)


def remote_picture(name):
    # Keep at most one original on disk, not the entire gallery.
    DOWNLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    target = DOWNLOAD_CACHE / 'current-image.part'
    try:
        url = f'{REMOTE_GALLERY}/download.php'
        with requests.get(url, params={'name': name}, stream=True, timeout=(10, 45),
                          headers={'User-Agent': 'GAGA-Indexer/3'}) as response:
            response.raise_for_status()
            size = 0
            with target.open('wb') as output:
                for chunk in response.iter_content(128 * 1024):
                    size += len(chunk)
                    if size > 35 * 1024 * 1024:
                        raise ValueError('Original photo exceeds 35 MB')
                    output.write(chunk)
        return decode(target)
    finally:
        target.unlink(missing_ok=True)


def decode(source):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(source) as picture:
                # JPEG draft reduces decoding memory before RGB/EXIF copies.
                if picture.width * picture.height > Image.MAX_IMAGE_PIXELS:
                    raise ValueError('Photo exceeds 40 megapixels')
                picture.draft('RGB', (1600, 1600))
                picture.thumbnail((1600, 1600))
                picture = ImageOps.exif_transpose(picture).convert('RGB')
                bgr = cv2.cvtColor(np.array(picture), cv2.COLOR_RGB2BGR)
                del picture
                return bgr
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
        # OpenCV detector/recognizer instances are mutable, not thread safe.
        with engine_lock:
            result = []
            for size in (800, 1600):
                scale = min(1.0, size / max(picture.shape[:2]))
                sample = cv2.resize(picture, None, fx=scale, fy=scale) if scale < 1 else picture
                self.detector.setInputSize((sample.shape[1], sample.shape[0]))
                _, faces = self.detector.detect(sample)
                for face in ([] if faces is None else faces):
                    aligned = self.recognizer.alignCrop(sample, face)
                    feature = self.recognizer.feature(aligned).flatten()
                    feature /= max(float(np.linalg.norm(feature)), 1e-12)
                    result.append(feature.tolist())
                if max(picture.shape[:2]) <= size:
                    break
            return result


def refresh():
    if not refresh_lock.acquire(blocking=False):
        return
    with lock:
        state.update(indexing=True, ready=False, processed=0, skipped=0, error=None,
                     stage='loading_gallery', last_photo_error=None)
    try:
        refresh_index()
    except Exception:
        with lock:
            state.update(ready=False, stage='error', error='Album indexing failed. Check Render logs.')
        raise
    finally:
        with lock:
            state['indexing'] = False
        refresh_lock.release()


def save_index(photos):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE.with_suffix('.tmp')
    temporary.write_text(json.dumps({'version': INDEX_VERSION,
                                    'source': GALLERY_SOURCE,
                                    'gallery': REMOTE_GALLERY,
                                    'photos': photos}))
    temporary.replace(CACHE)


def refresh_index():
    global records, engine
    files = gallery_files()
    if not files:
        raise ValueError('No gallery photos available')
    with lock:
        state.update(total=len(files), stage='loading_models', face_photos=0)
    if engine is None:
        engine = Faces()
    updated = {}
    for counter, (name, path) in enumerate(sorted(files.items()), 1):
        with lock:
            state.update(current_photo=name, stage='indexing')
        try:
            if path is None:
                stamp = ['remote', REMOTE_GALLERY, name, INDEX_VERSION]
            else:
                stat = path.stat()
                stamp = [stat.st_mtime_ns, stat.st_size]
            previous = records.get(name)
            if previous and previous.get('stamp') == stamp:
                entry = previous
            else:
                picture = remote_picture(name) if path is None else decode(path)
                try:
                    faces = engine.features(picture)
                finally:
                    del picture
                entry = {'stamp': stamp, 'faces': faces}
            updated[name] = entry
            with lock:
                records[name] = entry
                state['face_photos'] += bool(entry.get('faces'))
        except Exception as exc:
            logging.exception('Failed to index %s', name)
            with lock:
                state['skipped'] += 1
                state['last_photo_error'] = f'{name}: {type(exc).__name__}: {exc}'
        finally:
            with lock:
                state['processed'] = counter
            gc.collect()
        if counter % 25 == 0:
            save_index(updated)
            logging.info('Indexed %s/%s photos; %s contain faces',
                         counter, len(files), state['face_photos'])
    save_index(updated)
    with lock:
        records = updated
        complete = bool(updated) and not state['skipped']
        state.update(ready=complete, current_photo=None,
                     stage='ready' if complete else 'incomplete',
                     error=None if complete else 'Some photos failed to index; automatic retry is scheduled.')


def index_loop():
    global records
    try:
        if CACHE.exists():
            saved = json.loads(CACHE.read_text())
            if (saved.get('version') == INDEX_VERSION and saved.get('source') == GALLERY_SOURCE
                    and saved.get('gallery') == REMOTE_GALLERY):
                records = saved['photos']
                state.update(ready=False, total=len(records),
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
        time.sleep(300)


@app.route('/')
def home():
    return jsonify(
        service='RANI AI Face Matching Engine',
        status='active',
        build=BUILD,
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
    threading.Thread(target=refresh, daemon=True).start()
    return jsonify(message='Sync started in background', state=dict(state))


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

    expected = request.form.get('gallery_count', type=int) or 0
    with lock:
        complete = state['ready'] and not state['indexing'] and not state['error']
        records_snapshot = dict(records)
        total = state['total']
        skipped = state['skipped']
    if not complete or engine is None or not records_snapshot or (expected and expected != total):
        return jsonify(error='The photo library is still being prepared. Please try again shortly.',
                       indexing=state['indexing'], processed=state['processed'],
                       total=expected or total, stage=state['stage']), 503
    if skipped:
        return jsonify(error='Some event photos could not be indexed. Please ask the gallery owner to check the photo library.'), 503

    features = engine.features(picture)
    if len(features) == 0:
        return jsonify(error='No face detected in the photo. Please use a clear, front-facing selfie.'), 400

    query_vectors = [np.array(f, dtype=np.float32) for f in features]
    matches = []
    for name, entry in records_snapshot.items():
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
    del picture
    del features
    del query_vectors
    gc.collect()

    return jsonify(matches=matches, count=len(matches))


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-CSRF-Token, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    if '/api/' in request.path:
        response.headers['Cache-Control'] = 'no-store'
    return response


# Start background indexing loop
threading.Thread(target=index_loop, daemon=True, name='album-indexer').start()

if __name__ == '__main__':
    from waitress import serve
    port = int(os.getenv('PORT', '8093'))
    serve(app, host='0.0.0.0', port=port, threads=4)
