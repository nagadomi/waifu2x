#!/bin/zsh
# waifu2x daemon script
source /home/ubuntu/.zshrc
stdbuf -o 0 th web.lua  >> ./waifu2x.log 2>&1
