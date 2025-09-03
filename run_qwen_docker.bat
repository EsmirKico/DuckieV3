@echo off
echo 🚀 Starting Qwen 2.5 VL 7B with vLLM Docker Container
echo.

REM Pull the vLLM Docker image
echo Pulling vLLM Docker image...
docker pull vllm/vllm-openai:latest

REM Run Qwen 2.5 VL with vLLM
echo.
echo Starting Qwen 2.5 VL 7B server on port 8000...
echo Access will be available at: http://localhost:8000
echo API docs at: http://localhost:8000/docs
echo.

docker run --gpus all ^
    -v "%USERPROFILE%\.cache\huggingface:/root/.cache/huggingface" ^
    --env "HF_TOKEN=%HF_TOKEN%" ^
    -p 8000:8000 ^
    --ipc=host ^
    --name qwen-vl-server ^
    vllm/vllm-openai:latest ^
    --model Qwen/Qwen2.5-VL-7B-Instruct ^
    --chat-template template_qwen.jinja ^
    --max-model-len 8192 ^
    --limit-mm-per-prompt image=1 ^
    --host 0.0.0.0 ^
    --port 8000

echo.
echo Container stopped. To restart, run this script again.
pause 