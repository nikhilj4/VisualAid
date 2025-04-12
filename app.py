# app.py
from flask import Flask, request, render_template, jsonify
from PIL import Image
import easyocr
from transformers import pipeline
import torch
import re
from gtts import gTTS
import os
from werkzeug.utils import secure_filename
import logging
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disable multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Cache models
logger.info("Loading models at startup")
try:
    reader = easyocr.Reader(['en'], gpu=False)
    caption_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    llm = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    logger.info("Serving index page")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    logger.info("Received upload request")
    start_time = time.time()
    if 'file' not in request.files:
        logger.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Image saved: {filename}")
        
        try:
            # Process image
            logger.info("Opening image for processing")
            image = Image.open(filepath).convert('L')  # Grayscale
            image = image.resize((600, 600)) if image.size[0] > 600 else image
            
            # OCR with EasyOCR
            logger.info("Performing OCR with EasyOCR")
            ocr_results = reader.readtext(
                filepath, 
                detail=0, 
                contrast_ths=0.6, 
                adjust_contrast=0.9,
                allowlist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/=()[]{}.,: billion million "
            )
            ocr_text = " ".join(ocr_results)
            # Clean OCR errors
            ocr_text = re.sub(r"\bDaorld\b", "World", ocr_text)
            ocr_text = re.sub(r"\bAce\b", "Age", ocr_text)
            ocr_text = re.sub(r"Prospect=", "Prospects", ocr_text)
            ocr_text = re.sub(r"Under-255", "Under-25", ocr_text)
            ocr_text = re.sub(r"Under-155", "Under-15", ocr_text)
            ocr_text = re.sub(r"Under-5=", "Under-5", ocr_text)
            ocr_text = re.sub(r"Age=", "Age ", ocr_text)
            ocr_text = re.sub(r"\band\b", " and ", ocr_text)
            ocr_text = re.sub(r"billion billion", "billion", ocr_text)
            ocr_text = re.sub(r"[^\w\s\+\-\*/=^√÷×().%:, billion million]", "", ocr_text)
            logger.info(f"OCR output: {ocr_text}")
            
            # Image caption
            logger.info("Generating image caption")
            caption = caption_pipeline(image, max_new_tokens=100)[0]['generated_text']
            logger.info(f"Caption: {caption}")
            
            # Combine input for blind users
            combined_input = (
                f"Overview: {caption}\n"
                f"Details: {ocr_text.strip()}\n"
                "Start with the overview to paint a vivid picture of the image’s colors and shapes. "
                "Then, blend in key text details (like labels or numbers) to share what matters, fixing any errors. "
                "Adapt to any image type, keep it warm and clear, avoid repetitions, and tell a story in 50–100 words."
)

            logger.info(f"Combined input: {combined_input}")
            
            # LLM for explanation
            logger.info("Generating explanation with LLM")
            final_output = llm(
                combined_input, 
                min_length=50, 
                max_length=120, 
                max_new_tokens=100,
                do_sample=False,  # Avoid randomness
                num_beams=5       # Improve coherence
            )[0]['generated_text'].strip()
            logger.info(f"Explanation: {final_output}")
            
            # Generate audio
            logger.info("Generating audio")
            audio_text = final_output.replace(".", ". ")  # Add pauses
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'explanation.mp3')
            tts = gTTS(text=audio_text, slow=True)
            tts.save(audio_path)
            logger.info(f"Audio saved")
            
            logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
            return jsonify({
                'caption': caption,
                'ocr_text': ocr_text,
                'explanation': final_output,
                'image_url': f'/static/uploads/{filename}',
                'audio_url': '/static/uploads/explanation.mp3'
            })
        
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    logger.error("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=False)