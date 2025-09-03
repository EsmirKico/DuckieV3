#!/usr/bin/env python3
"""
Debug server for llama.cpp integration
"""

import uvicorn
import cv2
import numpy as np
import base64
import io
import json
import logging
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
import requests
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Debug Qwen Server")

QWEN_URL = "http://192.168.140.179:8080/completion"

@app.post("/test_drive")
async def test_drive(file: UploadFile = File(...)):
    """Minimal test endpoint"""
    try:
        logger.info("=== STARTING DEBUG TEST ===")
        
        # 1. Read image
        logger.info("1. Reading image...")
        image_data = await file.read()
        logger.info(f"Image data length: {len(image_data)}")
        
        # 2. Decode image  
        logger.info("2. Decoding image...")
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logger.info(f"Image shape: {image.shape if image is not None else 'None'}")
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # 3. Convert to base64
        logger.info("3. Converting to base64...")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        logger.info(f"Base64 length: {len(image_b64)}")
        
        # 4. Create simple prompt
        logger.info("4. Creating prompt...")
        prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<img src="data:image/jpeg;base64,{image_b64[:100]}...">
What do you see? Respond with: {{"description": "I see an image"}}<|im_end|>
<|im_start|>assistant"""
        
        logger.info(f"Prompt length: {len(prompt)}")
        
        # 5. Send to llama.cpp
        logger.info("5. Sending to llama.cpp...")
        payload = {
            "prompt": prompt,
            "n_predict": 50,
            "temperature": 0.1,
            "stop": ["<|im_end|>"],
            "stream": False
        }
        
        response = requests.post(
            QWEN_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"API error: {response.text}")
            raise HTTPException(status_code=500, detail=f"API error: {response.status_code}")
        
        result = response.json()
        content = result.get("content", "")
        logger.info(f"Response content: {content}")
        
        return {
            "status": "success",
            "content": content,
            "image_shape": image.shape,
            "base64_length": len(image_b64)
        }
        
    except Exception as e:
        logger.error(f"ERROR: {e}")
        logger.error(f"ERROR TYPE: {type(e)}")
        logger.error(f"TRACEBACK: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "qwen_url": QWEN_URL}

if __name__ == "__main__":
    print("🚀 Starting Debug Server on port 8003")
    uvicorn.run(app, host="192.168.140.144", port=8003) 