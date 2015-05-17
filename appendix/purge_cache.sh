#!/bin/zsh
# cron script for purge cache
source /home/ubuntu/.zshrc
cd /home/ubuntu/waifu2x
th appendix/purge_cache.lua
