#!/bin/bash

# 检查是否提供了参数
if [ -z "$1" ]; then
  echo "Usage: $0 <token>"
  exit 1
fi

echo "Start downloading models..."
# 获取token参数
TOKEN=$1
HF_TOKEN=$2
# 构建wget命令
wget -O ./models/checkpoints/architecturerealmix_v11.safetensors "https://civitai.com/api/download/models/431755?type=Model&format=SafeTensor&token=${TOKEN}" --content-disposition
wget -O ./models/loras/mjmimic.safetensors "https://civitai.com/api/download/models/283697?type=Model&format=SafeTensor&token=${TOKEN}" --content-disposition
wget -O ./models/controlnet/control_v11p_sd15_canny.safetensors https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/diffusion_pytorch_model.safetensors
wget -O ./models/checkpoints/XXMix_9realistic.safetensors "https://civitai.com/api/download/models/102222?type=Model&format=SafeTensor&token=${TOKEN}" --content-disposition
wget -O ./models/clip/clip_l.safetensors "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
wget -O ./models/clip/t5xxl_fp8_e4m3fn.safetensors "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
wget -O ./models/vae/ae.safetensors "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors"
wget --header="Authorization: Bearer ${HF_TOKEN}" -O ./models/checkpoints/flux1-dex.safetensors "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"


cd custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
pip install -r comfyui_controlnet_aux/requirements.txt