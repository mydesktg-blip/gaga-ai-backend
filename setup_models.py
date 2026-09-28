"""Download official OpenCV Zoo models and verify their Git LFS SHA256."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent / 'models'
MODELS = ['face_detection_yunet/face_detection_yunet_2023mar.onnx',
          'face_recognition_sface/face_recognition_sface_2021dec.onnx']

def setup():
    ROOT.mkdir(exist_ok=True)
    for model in MODELS:
        pointer = urlopen('https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/' + model, timeout=60).read().decode()
        digest = next(line.split('sha256:')[1] for line in pointer.splitlines() if line.startswith('oid '))
        path = ROOT / model.split('/')[-1]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        print('Downloading', path.name, flush=True)
        temporary = path.with_suffix('.download')
        with urlopen('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/' + model, timeout=120) as response, temporary.open('wb') as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
            temporary.unlink()
            raise RuntimeError('Model checksum mismatch')
        temporary.replace(path)
        license_url = 'https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/' + model.split('/')[0] + '/LICENSE'
        (ROOT / (path.stem + '.LICENSE')).write_bytes(urlopen(license_url, timeout=60).read())
    print('Models ready.')

if __name__ == '__main__':
    setup()
