#!/usr/bin/env python3
"""
Simple test for llama.cpp integration without images
"""

import requests
import json

API_PORT = 8000  # Using Qwen Docker server

def test_simple_text():
    """Test simple text completion"""
    print("🔍 Testing simple text completion...")
    
    prompt = """<|im_start|>system
You are an autonomous driving system for a DuckieBot.<|im_end|>
<|im_start|>user
Analyze a Duckietown road scene. Respond with JSON:
{
    "lane_detected": true,
    "lane_confidence": 0.8,
    "obstacle_detected": false,
    "driving_state": "lane_following"
}<|im_end|>
<|im_start|>assistant"""

    payload = {
        "prompt": prompt,
        "n_predict": 200,
        "temperature": 0.1,
        "stop": ["<|im_end|>"],
        "stream": False
    }
    
    try:
        response = requests.post(
            f"http://192.168.140.179:{API_PORT}/completion",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("content", "")
            print("✅ Text completion successful!")
            print(f"Response: {content}")
            
            # Try to extract JSON
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                try:
                    parsed = json.loads(json_content)
                    print("✅ JSON parsing successful!")
                    print(f"Parsed: {parsed}")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed: {e}")
            else:
                print("❌ No JSON found in response")
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_image_support():
    """Test if the server supports image input"""
    print("\n🔍 Testing image support...")
    
    # Create a simple base64 image (1x1 black pixel)
    import base64
    import io
    from PIL import Image
    import numpy as np
    
    # Create tiny test image
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    prompt = f"""<|im_start|>system
You are an autonomous driving system.<|im_end|>
<|im_start|>user
<img src="data:image/jpeg;base64,{img_b64}">
Analyze this image and respond with JSON:
{{"lane_detected": true, "driving_state": "lane_following"}}<|im_end|>
<|im_start|>assistant"""

    payload = {
        "prompt": prompt,
        "n_predict": 100,
        "temperature": 0.1,
        "stop": ["<|im_end|>"],
        "stream": False
    }
    
    try:
        response = requests.post(
            f"http://192.168.140.179:{API_PORT}/completion",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("content", "")
            print("✅ Image completion successful!")
            print(f"Response: {content[:200]}...")
        else:
            print(f"❌ Image request failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Testing llama.cpp Integration")
    print("=" * 40)
    
    test_simple_text()
    test_image_support()
    
    print("\n" + "=" * 40)
    print("🏁 Testing complete!") 