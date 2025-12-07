
from flask import Flask, request, jsonify, render_template
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
from io import BytesIO

# ----------------------------------------------------------------------
# 1. Initialization and Model Loading
# ----------------------------------------------------------------------
app = Flask(__name__)
# NOTE: Replace 'model_dir' with the actual path to your saved ViT model and processor files.
MODEL_DIR = "model_dir" 

# Load the pre-trained/fine-tuned Vision Transformer (ViT) model and processor
try:
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR) 
    model = ViTForImageClassification.from_pretrained(MODEL_DIR)
except Exception as e:
    # A robust check in a real application would handle this gracefully
    print(f"Error loading model components from {MODEL_DIR}: {e}")
    # Consider exiting or setting a flag if model fails to load

# ----------------------------------------------------------------------
# 2. Routes
# ----------------------------------------------------------------------

# Root route to render the front-end HTML (index.html) [03:56:22]
@app.route('/')
def index():
    return render_template('index.html') 

# POST API route to handle image prediction [03:56:39]
@app.route('/predict', methods=['POST'])
def predict():
    # Check if 'image' file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in the request'}), 400

    file = request.files['image']

    try:
        # Open the image file from the stream
        image = Image.open(file.stream) 
        
        # Convert image to RGB format, as specified in the walkthrough [04:18:00]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process the image into tensors using the ViTImageProcessor [04:21:00]
        inputs = processor(images=image, return_tensors="pt") 
    
    except Exception as e:
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500

    # Perform prediction without tracking gradients (for inference)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits # Get the raw output scores [04:29:00]
    
    # Get the predicted class ID (0 or 1) and calculate confidence
    predicted_class_id = logits.argmax().item()
    
    # Use Softmax to convert logits into probabilities (confidence scores)
    probabilities = torch.softmax(logits, dim=1)
    confidence = probabilities[0][predicted_class_id].item()

    # Map the class ID back to the human-readable label ("Real" or "Fake") [04:31:00]
    if hasattr(model.config, 'id2label'):
        predicted_label = model.config.id2label[predicted_class_id]
    else:
        # Fallback based on typical binary classification mapping used in the training notebook
        predicted_label = "Real" if predicted_class_id == 0 else "Fake"

    # Return the prediction and confidence as a JSON response
    return jsonify({
        'prediction': predicted_label,
        'confidence': round(confidence, 4) # Round confidence for cleaner display
    })

# ----------------------------------------------------------------------
# 3. Main Execution
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Running in debug mode is useful during development
    app.run(debug=True)