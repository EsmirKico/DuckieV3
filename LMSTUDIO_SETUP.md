# 🚗 DuckieBot Autonomous Driving with LM Studio + Qwen

## Quick Setup with Your Existing LM Studio

Since you already have Qwen2.5-VL-7B running in LM Studio at `http://192.168.140.1:1234`, here's the simplified setup:

## 🚀 Server Setup (Super Simple!)

### 1. Install Dependencies (Much Lighter!)
```bash
cd /home/lesgrossman/ducky/duckie_v2
pip install -r server/lmstudio_requirements.txt
```

### 2. Start the Autonomous Server
```bash
python3 server/autonomous_server_lmstudio.py
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
python3 server/test_lmstudio_api.py
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

### 📊 **Performance Expectations with Qwen2.5-VL-7B:**
- **Response Time**: ~3-7 seconds per image (7B model is more thorough)
- **GPU Usage**: Managed by LM Studio
- **Accuracy**: Excellent (7B model provides superior vision analysis)
- **Stability**: More stable than loading models directly

## 🔧 LM Studio Configuration

### **Current Model Configuration:**
- **Model**: `qwen/qwen2.5-vl-7b` ✅ (configured)
- **Context Length**: 4096 tokens
- **Temperature**: 0.1 (for consistent responses)
- **Max Tokens**: 1000

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
python3 server/test_lmstudio_api.py
```

### **Expected Test Output:**
```
🔗 Testing direct LM Studio connection...
✅ LM Studio connected! Models available: 1
   📦 qwen/qwen2.5-vl-7b

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
  - Verify model name is exactly `qwen/qwen2.5-vl-7b` in LM Studio
  - Check model is fully loaded and ready

### **Performance Tuning for 7B Model:**
- **Slow response (>10 seconds)**: 
  - Check GPU memory usage in LM Studio
  - Reduce image resolution in autonomous_detector.py
  - Consider processing every 3rd frame instead of every frame

### **API Errors:**
- **Timeout errors**: Increase timeout to 45 seconds in autonomous_detector.py
- **JSON parsing errors**: 7B model should provide better structured responses

## 🔄 Switching Between Versions

### **Tennis Ball Following** (Original):
```bash
./run_object_detection.sh
```

### **Autonomous (Direct Qwen)**:
```bash
python3 server/autonomous_server.py  # Heavy version
```

### **Autonomous (LM Studio)** - RECOMMENDED:
```bash
python3 server/autonomous_server_lmstudio.py  # Your version!
```

## 📈 Next Steps

1. **Start with LM Studio version** (what you have)
2. **Test with webcam** using test script
3. **Deploy to DuckieBot** once server works
4. **Fine-tune parameters** for your specific track

## 🎯 Why Qwen2.5-VL-7B is Excellent for This

- **Superior accuracy** for lane detection ✅
- **Better obstacle recognition** ✅  
- **More reliable JSON responses** ✅
- **Excellent spatial reasoning** ✅
- **Robust in various lighting** ✅

**Trade-off**: Slower processing (~3-7 seconds) but much higher accuracy

## 💡 Performance Tips for 7B Model

### **Optimize for Speed:**
```python
# In autonomous_detector.py, increase processing interval:
self.min_process_interval = 0.3  # Process every 3rd frame (3 FPS)
```

### **Optimize for Accuracy:**
```python
# In autonomous_detector.py, increase API timeout:
self.api_timeout = rospy.get_param('~api_timeout', 10.0)  # 10 second timeout
```

---

**Your DuckieBot + LM Studio + Qwen2.5-VL-7B = Premium Autonomous Driving! 🚀** 