from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
import logging
from typing import Dict, Any
import uvicorn
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Tomato Disease Classifier API",
    description="AI-powered tomato plant disease detection system",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
TOMATO_CLASSES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Disease information and recommendations
DISEASE_INFO = {
    'Tomato___Bacterial_spot': {
        'name': 'Bacterial Spot',
        'description': 'A bacterial disease causing dark spots on leaves and fruit, leading to defoliation and reduced yield.',
        'severity': 'High',
        'recommendations': [
            'Remove affected plant parts immediately',
            'Apply copper-based bactericides',
            'Improve air circulation around plants',
            'Avoid overhead watering',
            'Use disease-resistant varieties'
        ]
    },
    'Tomato___Early_blight': {
        'name': 'Early Blight',
        'description': 'A fungal disease causing brown spots with concentric rings on lower leaves.',
        'severity': 'Medium',
        'recommendations': [
            'Remove lower infected leaves',
            'Apply fungicide containing chlorothalonil',
            'Mulch around plants to prevent soil splashing',
            'Ensure proper spacing for air circulation',
            'Water at soil level to keep leaves dry'
        ]
    },
    'Tomato___Late_blight': {
        'name': 'Late Blight',
        'description': 'A serious fungal disease that can destroy entire crops. Shows as dark, water-soaked lesions.',
        'severity': 'Critical',
        'recommendations': [
            'Remove and destroy infected plants immediately',
            'Apply preventive fungicide treatments',
            'Ensure excellent air circulation',
            'Avoid watering in the evening',
            'Consider resistant varieties for future planting'
        ]
    },
    'Tomato___Leaf_Mold': {
        'name': 'Leaf Mold',
        'description': 'A fungal disease common in humid conditions, causing yellow patches that turn brown.',
        'severity': 'Medium',
        'recommendations': [
            'Improve greenhouse ventilation',
            'Reduce humidity levels',
            'Remove affected leaves',
            'Apply appropriate fungicides',
            'Space plants adequately'
        ]
    },
    'Tomato___Septoria_leaf_spot': {
        'name': 'Septoria Leaf Spot',
        'description': 'A fungal disease causing small, circular spots with dark borders on leaves.',
        'severity': 'Medium',
        'recommendations': [
            'Remove infected lower leaves',
            'Apply fungicide treatments',
            'Mulch to prevent soil splashing',
            'Water at the base of plants',
            'Rotate crops annually'
        ]
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'name': 'Spider Mites',
        'description': 'Tiny pests that cause stippling, yellowing, and webbing on leaves.',
        'severity': 'Medium',
        'recommendations': [
            'Increase humidity around plants',
            'Use predatory mites as biological control',
            'Apply miticide if infestation is severe',
            'Remove heavily infested leaves',
            'Avoid over-fertilizing with nitrogen'
        ]
    },
    'Tomato___Target_Spot': {
        'name': 'Target Spot',
        'description': 'A fungal disease causing circular lesions with concentric rings.',
        'severity': 'Medium',
        'recommendations': [
            'Remove affected plant debris',
            'Apply preventive fungicide sprays',
            'Ensure good air circulation',
            'Practice crop rotation',
            'Water at soil level'
        ]
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'name': 'Yellow Leaf Curl Virus',
        'description': 'A viral disease causing upward curling and yellowing of leaves.',
        'severity': 'High',
        'recommendations': [
            'Remove infected plants immediately',
            'Control whitefly vectors',
            'Use reflective mulches',
            'Plant virus-resistant varieties',
            'Maintain weed-free growing areas'
        ]
    },
    'Tomato___Tomato_mosaic_virus': {
        'name': 'Mosaic Virus',
        'description': 'A viral disease causing mottled light and dark green patterns on leaves.',
        'severity': 'High',
        'recommendations': [
            'Remove and destroy infected plants',
            'Disinfect tools between plants',
            'Control aphid and thrips vectors',
            'Use certified disease-free seeds',
            'Practice good sanitation'
        ]
    },
    'Tomato___healthy': {
        'name': 'Healthy',
        'description': 'Your tomato plant appears healthy with no signs of disease.',
        'severity': 'None',
        'recommendations': [
            'Continue current care routine',
            'Monitor regularly for any changes',
            'Maintain consistent watering schedule',
            'Ensure adequate nutrition',
            'Practice preventive disease management'
        ]
    }
}

# Global variables for model
model = None
device = None
transform = None

def load_model():
    """Load the trained tomato disease classification model"""
    global model, device, transform
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize model architecture (same as training)
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(TOMATO_CLASSES))
        )
        
        # Load trained weights
        model_path = "best_tomato_model.pth"
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Define image preprocessing (same as validation transform)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess uploaded image for model inference"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        return image_tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def get_prediction(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Get model prediction for preprocessed image"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top prediction
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = TOMATO_CLASSES[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            
            # Get disease information
            disease_info = DISEASE_INFO.get(predicted_class, {})
            
            return {
                'disease': disease_info.get('name', predicted_class),
                'confidence': round(confidence_score, 2),
                'description': disease_info.get('description', 'No description available'),
                'severity': disease_info.get('severity', 'Unknown'),
                'recommendations': disease_info.get('recommendations', []),
                'raw_class': predicted_class
            }
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Tomato Disease Classifier API...")
    if not load_model():
        logger.error("Failed to load model. API may not function properly.")
    else:
        logger.info("API ready to serve predictions!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Tomato Disease Classifier API",
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "supported_classes": len(TOMATO_CLASSES),
        "classes": TOMATO_CLASSES
    }

@app.post("/classify")
async def classify_tomato_disease(file: UploadFile = File(...)):
    """
    Classify tomato disease from uploaded image
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        
    Returns:
        JSON response with disease classification and recommendations
    """
    
    # Validate model is loaded
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please try again later."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    # Validate file size (limit to 10MB)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File size too large. Maximum size is 10MB."
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        logger.info(f"Processing image: {file.filename}, size: {len(image_bytes)} bytes")
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Get prediction
        result = get_prediction(image_tensor)
        
        logger.info(f"Prediction: {result['disease']} ({result['confidence']:.2f}%)")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": result,
                "message": "Image classified successfully"
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error during classification: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image classification"
        )

@app.get("/diseases")
async def get_disease_info():
    """Get information about all supported diseases"""
    return {
        "success": True,
        "data": {
            "total_classes": len(TOMATO_CLASSES),
            "diseases": DISEASE_INFO
        }
    }

@app.get("/diseases/{disease_name}")
async def get_specific_disease_info(disease_name: str):
    """Get information about a specific disease"""
    
    # Find matching disease (case insensitive)
    matching_disease = None
    for class_name, info in DISEASE_INFO.items():
        if (disease_name.lower() in class_name.lower() or 
            disease_name.lower() in info.get('name', '').lower()):
            matching_disease = {
                "class_name": class_name,
                **info
            }
            break
    
    if not matching_disease:
        raise HTTPException(
            status_code=404,
            detail=f"Disease '{disease_name}' not found"
        )
    
    return {
        "success": True,
        "data": matching_disease
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "status_code": 500
        }
    )

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",  # Assuming this file is named main.py
        host="0.0.0.0",
        port=8000,
        reload=True,  # Remove in production
        log_level="info"
    )