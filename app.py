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
THRESHOLD = float(os.getenv('FACE_THRESHOLD', '0.38'))
BUILD = 'gaga-index-v6'
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
    logging.info('Fetching gallery list from %s', REMOTE_GALLERY)
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
    logging.info('Gallery list loaded: %s photos', len(names))
    return dict.fromkeys(names)


def remote_picture(name):
    DOWNLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    target = DOWNLOAD_CACHE / f'{name}.part'
    urls = [
        f'{REMOTE_GALLERY}/thumbs/{name}',
        f'{REMOTE_GALLERY}/thumb.php?name={name}',
        f'{REMOTE_GALLERY}/download.php?name={name}'
    ]
    for url in urls:
        for attempt in range(3):
            try:
                with requests.get(url, stream=True, timeout=(10, 45),
                                  headers={'User-Agent': 'GAGA-Indexer/3'}) as response:
                    if response.status_code == 200:
                        size = 0
                        with target.open('wb') as output:
                            for chunk in response.iter_content(128 * 1024):
                                size += len(chunk)
                                if size > 35 * 1024 * 1024:
                                    raise ValueError('Photo exceeds 35 MB')
                                output.write(chunk)
                        img = decode(target)
                        return img
            except Exception:
                if attempt < 2:
                    time.sleep(1)
            finally:
                target.unlink(missing_ok=True)
    raise ValueError(f'Could not download photo {name}')


def decode(source):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(source) as picture:
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
        has_existing = bool(records)
        state.update(indexing=True, ready=has_existing, processed=0, skipped=0,
                     stage='loading_gallery', last_photo_error=None)
    try:
        refresh_index()
    except Exception as exc:
        logging.exception('Album indexing failed: %s', exc)
        with lock:
            if not records:
                state.update(ready=False, stage='error',
                             error=f'Album indexing failed: {type(exc).__name__}. Check Render logs.')
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


def load_cache():
    global records, engine
    # 1. Try local CACHE
    try:
        if CACHE.exists():
            saved = json.loads(CACHE.read_text())
            if saved.get('version') == INDEX_VERSION and saved.get('photos'):
                records = saved['photos']
                with lock:
                    state.update(
                        ready=True,
                        total=len(records),
                        processed=len(records),
                        face_photos=sum(bool(item.get('faces')) for item in records.values()),
                        stage='ready',
                        error=None
                    )
                logging.info('Loaded %d cached photos from %s', len(records), CACHE)
                if engine is None:
                    engine = Faces()
                return True
    except Exception as exc:
        logging.warning('Could not load local cache: %s', exc)

    # 2. Try root faces.json if CACHE does not exist
    root_faces = BASE / 'faces.json'
    if root_faces.exists():
        try:
            saved = json.loads(root_faces.read_text())
            if saved.get('version') == INDEX_VERSION and saved.get('photos'):
                records = saved['photos']
                save_index(records)
                with lock:
                    state.update(
                        ready=True,
                        total=len(records),
                        processed=len(records),
                        face_photos=sum(bool(item.get('faces')) for item in records.values()),
                        stage='ready',
                        error=None
                    )
                logging.info('Loaded %d photos from %s', len(records), root_faces)
                if engine is None:
                    engine = Faces()
                return True
        except Exception as exc:
            logging.warning('Could not load root faces.json: %s', exc)

    return False


def refresh_index():
    global records, engine

    # Check remote faces.json / get_faces.php first for instant 1-second sync
    if GALLERY_SOURCE == 'remote':
        sync_urls = [
            f'{REMOTE_GALLERY}/get_faces.php',
            f'{REMOTE_GALLERY}/faces.json'
        ]
        for sync_url in sync_urls:
            try:
                r = requests.get(sync_url, timeout=(10, 45),
                                 headers={'User-Agent': 'GAGA-Indexer/3'})
                if r.status_code == 200:
                    data = r.json()
                    if data.get('version') == INDEX_VERSION and data.get('photos'):
                        remote_photos = data['photos']
                        if len(remote_photos) >= len(records):
                            records = remote_photos
                            save_index(records)
                            if engine is None:
                                engine = Faces()
                            with lock:
                                state.update(
                                    ready=True,
                                    total=len(records),
                                    processed=len(records),
                                    face_photos=sum(bool(item.get('faces')) for item in records.values()),
                                    stage='ready',
                                    error=None,
                                    skipped=0
                                )
                            logging.info('Instantly synced %d photos from %s!', len(records), sync_url)
                            return
            except Exception as e:
                logging.info('Remote sync from %s failed (%s), trying next...', sync_url, e)

    files = gallery_files()
    if not files:
        if not records:
            raise ValueError('No gallery photos available')
        return

    with lock:
        state.update(total=len(files), stage='loading_models')

    if engine is None:
        engine = Faces()

    updated = dict(records)
    skipped_count = 0

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
        except Exception as exc:
            logging.warning('Could not index %s: %s', name, exc)
            skipped_count += 1
            with lock:
                state['skipped'] = skipped_count
                state['last_photo_error'] = f'{name}: {type(exc).__name__}: {exc}'
        finally:
            with lock:
                state['processed'] = counter
                state['face_photos'] = sum(bool(item.get('faces')) for item in updated.values())
            gc.collect()

        if counter % 50 == 0:
            save_index(updated)
            with lock:
                records = updated
                state['ready'] = bool(updated)

    save_index(updated)
    with lock:
        records = updated
        complete = bool(updated)
        state.update(ready=complete, current_photo=None,
                     stage='ready' if complete else 'incomplete',
                     error=None)


def index_loop():
    load_cache()
    while True:
        try:
            refresh()
        except Exception as e:
            logging.exception('Album indexing failed: %s', e)
        time.sleep(300)


@app.route('/')
def home():
    return jsonify(
        service='RANI AI Face Matching Engine',
        status='active',
        build=BUILD,
        ready=bool(records),
        total_photos=len(records),
        face_photos=sum(bool(item.get('faces')) for item in records.values())
    )


@app.route('/api/status')
@app.route('/gaga_collection/api/status')
def status():
    with lock:
        st = dict(state)
        st['total'] = len(records) or st['total']
        st['face_photos'] = sum(bool(item.get('faces')) for item in records.values())
        st['ready'] = bool(records)
        return jsonify(st)


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

    global engine
    if engine is None:
        try:
            engine = Faces()
        except Exception as exc:
            logging.exception('Engine init error: %s', exc)
            return jsonify(error='AI engine initializing. Please try again in a few moments.'), 503

    with lock:
        records_snapshot = dict(records)

    if not records_snapshot:
        return jsonify(error='The photo library is still being prepared. Please try again shortly.',
                       indexing=state['indexing'], processed=state['processed'],
                       total=state['total'], stage=state['stage']), 503

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


indexer_start_lock = threading.Lock()
indexer_thread = None
indexer_pid = None


def start_indexer():
    global indexer_thread, indexer_pid
    with indexer_start_lock:
        pid = os.getpid()
        if indexer_pid == pid and indexer_thread is not None and indexer_thread.is_alive():
            return
        load_cache()
        indexer_thread = threading.Thread(target=index_loop, daemon=True, name='album-indexer')
        indexer_pid = pid
        state['worker_pid'] = pid
        logging.info('Starting %s indexer in serving process %s', BUILD, pid)
        indexer_thread.start()


@app.before_request
def start_worker_indexer():
    start_indexer()

if __name__ == '__main__':
    from waitress import serve
    start_indexer()
    port = int(os.getenv('PORT', '8093'))
    serve(app, host='0.0.0.0', port=port, threads=4)
