#!/bin/sh -x

# maybe you should backup models

rebuild() {
    if [ -f models/${1}/scale2.0x_model.t7 ] && [ ! -h models/${1}/scale2.0x_model.t7 ] ; then
	th tools/rebuild_model.lua -i models/${1}/scale2.0x_model.t7 -o models/${1}/scale2.0x_model.t7 -backend cunn -model $2
    fi
    if [ -f models/${1}/noise1_model.t7 ] && [ ! -h models/${1}/noise1_model.t7 ]; then
	th tools/rebuild_model.lua -i models/${1}/noise1_model.t7 -o models/${1}/noise1_model.t7 -backend cunn -model $2
    fi
    if [ -f models/${1}/noise2_model.t7 ] && [ ! -h models/${1}/noise2_model.t7 ]; then
	th tools/rebuild_model.lua -i models/${1}/noise2_model.t7 -o models/${1}/noise2_model.t7 -backend cunn -model $2
    fi
    if [ -f models/${1}/noise3_model.t7 ] && [ ! -h models/${1}/noise3_model.t7 ]; then
	th tools/rebuild_model.lua -i models/${1}/noise3_model.t7 -o models/${1}/noise3_model.t7 -backend cunn -model $2
    fi
}
rebuild vgg_7/art vgg_7
rebuild vgg_7/art_y vgg_7
rebuild vgg_7/photo vgg_7
rebuild vgg_7/ukbench vgg_7
rebuild upconv_7/art upconv_7
