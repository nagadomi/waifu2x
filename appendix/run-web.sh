#!/bin/zsh
# waifu2x daemon script
gpu=1
port=8812
if [ $# -eq 2 ]; then
    gpu=$1
    port=$2
fi
source /home/ubuntu/.zshrc
echo stdbuf -o 0 th web.lua -gpu $gpu -port $port >> ./waifu2x_${port}.log 2>&1
stdbuf -o 0 th web.lua -gpu $gpu -port $port >> ./waifu2x_${port}.log 2>&1
