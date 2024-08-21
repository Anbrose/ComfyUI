#!/bin/bash

# 检查是否提供了参数
if [ -z "$1" ]; then
  echo "Usage: $0 <token>"
  exit 1
fi

echo "Start downloading models..."
# 获取token参数
TOKEN=$1
# 构建wget命令
wget -O ./models/checkpoints/architecturerealmix_v11.safetensors "https://civitai.com/api/download/models/431755?type=Model&format=SafeTensor&token=${TOKEN}" --content-disposition
wget -O ./models/loras/mjmimic.safetensors "https://civitai.com/api/download/models/283697?type=Model&format=SafeTensor&token=${TOKEN}" --content-disposition
wget -O ./models/controlnet/control_v11p_sd15_canny.safetensors https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/diffusion_pytorch_model.safetensors

cd custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
pip install -r comfyui_controlnet_aux/requirements.txt