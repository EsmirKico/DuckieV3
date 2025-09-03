# 🚗 DuckieBot Autonomous Driving with LM Studio + Qwen

## Quick Setup with Your Existing LM Studio

Since you already have Qwen running in LM Studio at `http://192.168.140.1:1234`, here's the simplified setup:

## 🚀 Server Setup (Super Simple!)

### 1. Install Dependencies (Much Lighter!)
```bash
cd /home/lesgrossman/ducky/duckie_v2
pip install -r server/lmstudio_requirements.txt
```

### 2. Start the Autonomous Server
```bash
python server/autonomous_server_lmstudio.py
```

Expected output:
```
🚀 Using LM Studio for Qwen Vision Model
📡 LM Studio URL: http://192.168.140.1:1234
✅ LM Studio connected! Available models: 1
✅ LM Studio connection verified!
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 3. Test the Integration
```bash
python server/test_lmstudio_api.py
```

## 🤖 DuckieBot Setup (Same as Before)

1. **Update IP in autonomous_detector.py** (already done: `192.168.140.144`)
2. **Deploy to DuckieBot**: Upload the code and run `./run_autonomous_driving.sh`

## 🎯 Key Advantages of LM Studio Version

### ✅ **Pros:**
- **Much faster setup** - No model downloading/loading
- **Less GPU memory** - LM Studio handles model optimization
- **Easy model switching** - Change models in LM Studio UI
- **Better performance** - LM Studio optimizations
- **Shared resources** - Use same LM Studio for other tasks

### 📊 **Performance Expectations:**
- **Response Time**: ~2-5 seconds per image (depends on model size)
- **GPU Usage**: Managed by LM Studio
- **Accuracy**: Same as direct Qwen2.5-VL usage
- **Stability**: More stable than loading models directly

## 🔧 LM Studio Configuration

### **Recommended Settings in LM Studio:**
1. **Model**: Qwen2-VL-3B or Qwen2-VL-7B
2. **Context Length**: 4096 tokens
3. **Temperature**: 0.1 (for consistent responses)
4. **Max Tokens**: 1000

### **If Using Different Model Name:**
Edit `server/autonomous_server_lmstudio.py` line 170:
```python
"model": "your-model-name-here",  # Update this to match LM Studio
```

## 🧪 Testing

### **Test LM Studio Connection:**
```bash
curl http://192.168.140.1:1234/v1/models
```

### **Test Full Pipeline:**
```bash
python server/test_lmstudio_api.py
```

### **Expected Test Output:**
```
🔗 Testing direct LM Studio connection...
✅ LM Studio connected! Models available: 1
   📦 qwen2-vl-3b-instruct

✅ Autonomous server is running!
✅ LM Studio integration working!

🤖 LM STUDIO AUTONOMOUS DRIVING RESPONSE:
Lane Offset: -0.234
Lane Confidence: 0.87
Steering: -0.187 (LEFT)
Speed: 0.320 (FORWARD)
State: LANE_FOLLOWING
```

## 🚨 Troubleshooting

### **LM Studio Issues:**
- **"Cannot connect to LM Studio"**:
  - Check LM Studio is running on port 1234
  - Verify network connectivity: `ping 192.168.140.1`
  - Try restarting LM Studio

### **Model Issues:**
- **"Model not found"**: 
  - Check model name in LM Studio
  - Update model name in `autonomous_server_lmstudio.py`

### **Slow Response:**
- **Use smaller model** (Qwen2-VL-3B instead of 7B)
- **Reduce image resolution** in autonomous_detector.py
- **Check GPU utilization** in LM Studio

### **API Errors:**
- **Timeout errors**: Increase timeout in autonomous_detector.py
- **JSON parsing errors**: Check LM Studio model supports vision input

## 🔄 Switching Between Versions

### **Tennis Ball Following** (Original):
```bash
./run_object_detection.sh
```

### **Autonomous (Direct Qwen)**:
```bash
python server/autonomous_server.py  # Heavy version
```

### **Autonomous (LM Studio)** - RECOMMENDED:
```bash
python server/autonomous_server_lmstudio.py  # Your version!
```

## 📈 Next Steps

1. **Start with LM Studio version** (what you have)
2. **Test with webcam** using test script
3. **Deploy to DuckieBot** once server works
4. **Fine-tune parameters** for your specific track

## 🎯 Why LM Studio is Perfect for This

- **You already have it running** ✅
- **Optimized for your hardware** ✅  
- **Easy to manage models** ✅
- **Shared with other projects** ✅
- **Better performance** ✅

---

**Your DuckieBot + LM Studio + Qwen = Perfect Autonomous Driving! 🚀** 