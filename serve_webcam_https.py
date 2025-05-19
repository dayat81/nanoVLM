import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow/MediaPipe INFO and WARNING
class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass
sys.stderr = DevNull()
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, send_from_directory, request, send_file, jsonify
import io
from PIL import Image
import numpy as np
import cv2
import torch
import pickle
import base64
from models.vision_language_model import VisionLanguageModel
from data.processors import get_image_processor, get_tokenizer

app = Flask(__name__)

# Load nanoVLM model and image processor at startup
model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-222M")
model.eval()
image_processor = get_image_processor(model.cfg.vit_img_size)
DB_PATH = "person_db.pkl"

@app.route('/')
def index():
    return send_from_directory('.', 'webcam.html')

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('.', filename)

def extract_embedding(image_pil):
    image_tensor = image_processor(image_pil).unsqueeze(0)
    with torch.no_grad():
        emb = model.vision_encoder(image_tensor)
        if hasattr(emb, 'cpu'):
            emb = emb.cpu()
        return emb.flatten().numpy()

def l2_normalize(x):
    return x / np.linalg.norm(x)

def load_db():
    try:
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

def save_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

@app.route('/classify_person', methods=['POST'])
def classify_person():
    file = request.files['image']
    name = request.form.get('name', None)
    image = Image.open(file.stream).convert('RGB')
    img_np = np.array(image)
    h, w, _ = img_np.shape
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, fw, fh = faces[0]
        cx = x + fw // 2
        cy = y + fh // 2
    else:
        cx, cy = w // 2, h // 2
    crop_size = 224
    x1 = max(0, cx - crop_size // 2)
    y1 = max(0, cy - crop_size // 2)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)
    if x2 - x1 < crop_size:
        x1 = max(0, x2 - crop_size)
    if y2 - y1 < crop_size:
        y1 = max(0, y2 - crop_size)
    crop = img_np[y1:y2, x1:x2]
    crop_pil = Image.fromarray(crop).resize((224, 224))
    buf = io.BytesIO()
    crop_pil.save(buf, format='PNG')
    buf.seek(0)
    cropped_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    # Encode full image
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    full_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    emb = l2_normalize(extract_embedding(crop_pil))
    db = load_db()

    # Always generate description for the original image
    def generate_description(image_pil):
        tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
        text = "Describe this image in as much detail as possible. Do not stop until you have described everything."
        template = f"Question: {text} Answer:"
        encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
        tokens = encoded_batch['input_ids']
        image_tensor = image_processor(image_pil).unsqueeze(0)
        with torch.no_grad():
            gen = model.generate(tokens, image_tensor, max_new_tokens=256)
            desc_raw = tokenizer.batch_decode(gen, skip_special_tokens=False)[0]
            print("Raw output:", desc_raw)
            return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    description = generate_description(image)

    if name:
        db.append((emb, name))
        save_db(db)
        return jsonify({
            "status": "added", 
            "name": name, 
            "cropped_image": cropped_b64,
            "full_image": full_b64,
            "similarity": None, 
            "description": description
        })
    else:
        if not db:
            return jsonify({"status": "unknown", "reason": "DB empty", "cropped_image": cropped_b64, "full_image": full_b64, "similarity": None, "description": description})
        embs, names = zip(*db)
        embs = np.stack([l2_normalize(e) for e in embs])
        dists = np.linalg.norm(embs - emb, axis=1)
        idx = np.argmin(dists)
        if dists[idx] < 1.0:
            return jsonify({"status": "recognized", "name": names[idx], "cropped_image": cropped_b64, "full_image": full_b64, "similarity": float(dists[idx]), "description": description})
        else:
            return jsonify({"status": "best_guess", "name": names[idx], "cropped_image": cropped_b64, "full_image": full_b64, "similarity": float(dists[idx]), "description": description})

@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        img_np = np.array(image)
        h, w, _ = img_np.shape
        try:
            import mediapipe as mp
        except ImportError:
            return jsonify({'found': False, 'error': 'mediapipe not installed'}), 500
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(img_np)
            if results.detections:
                detection = max(results.detections, key=lambda d: d.score[0])
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                return jsonify({'found': True, 'x': x, 'y': y, 'w': bw, 'h': bh})
            else:
                return jsonify({'found': False})
    except Exception as e:
        print("Error in /detect_face:", e)
        return jsonify({'found': False, 'error': str(e)}), 500

@app.route('/describe_image', methods=['POST'])
def describe_image():
    file = request.files['image']
    prompt = request.form.get('prompt', 'what is the person doing?')
    image = Image.open(file.stream).convert('RGB')
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    template = f"Question: {prompt} Answer:"
    encoded_batch = tokenizer.batch_encode_plus([template], return_tensors="pt")
    tokens = encoded_batch['input_ids']
    image_tensor = image_processor(image).unsqueeze(0)
    with torch.no_grad():
        gen = model.generate(tokens, image_tensor, max_new_tokens=30)
        description = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return jsonify({"description": description})

if __name__ == '__main__':
    # Make sure cert.pem and key.pem exist in the same directory
    cert_file = os.path.join(os.path.dirname(__file__), 'cert.pem')
    key_file = os.path.join(os.path.dirname(__file__), 'key.pem')
    if not (os.path.exists(cert_file) and os.path.exists(key_file)):
        print('Please generate cert.pem and key.pem using:')
        print('  openssl req -new -x509 -keyout key.pem -out cert.pem -days 365 -nodes')
        exit(1)
    app.run(host='0.0.0.0', port=443, ssl_context=(cert_file, key_file)) 