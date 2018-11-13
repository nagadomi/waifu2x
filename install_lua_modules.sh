#!/bin/bash

CUDNN_BRANCH=R7 # for cudnn7
CUDNN_WORK_DIR=.cudnn

install_cudnn()
{
    rm -fr $CUDNN_WORK_DIR
    git clone https://github.com/soumith/cudnn.torch.git -b $CUDNN_BRANCH $CUDNN_WORK_DIR
    cd $CUDNN_WORK_DIR
    luarocks make cudnn-scm-1.rockspec
    cd ..
    rm -fr $CUDNN_WORK_DIR
}

luarocks install graphicsmagick
luarocks install lua-csnappy
luarocks install md5
luarocks install uuid
luarocks install csvigo
install_cudnn
PREFIX=$HOME/torch/install luarocks install turbo
