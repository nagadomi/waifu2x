#!/bin/sh -x

export_model() {
    if [ -f models/${1}/scale2.0x_model.t7 ] && [ ! -h models/${1}/scale2.0x_model.t7 ] ; then
	th tools/export_model.lua -i models/${1}/scale2.0x_model.t7 -o models/${1}/scale2.0x_model.json
    fi

    if [ -f models/${1}/noise0_model.t7 ] && [ ! -h models/${1}/noise0_model.t7 ]; then
	th tools/export_model.lua -i models/${1}/noise0_model.t7 -o models/${1}/noise0_model.json
    fi
    if [ -f models/${1}/noise1_model.t7 ] && [ ! -h models/${1}/noise1_model.t7 ]; then
	th tools/export_model.lua -i models/${1}/noise1_model.t7 -o models/${1}/noise1_model.json
    fi
    if [ -f models/${1}/noise2_model.t7 ] && [ ! -h models/${1}/noise2_model.t7 ]; then
	th tools/export_model.lua -i models/${1}/noise2_model.t7 -o models/${1}/noise2_model.json
    fi
    if [ -f models/${1}/noise3_model.t7 ] && [ ! -h models/${1}/noise3_model.t7 ]; then
	th tools/export_model.lua -i models/${1}/noise3_model.t7 -o models/${1}/noise3_model.json
    fi

    if [ -f models/${1}/noise0_scale2.0x_model.t7 ] && [ ! -h models/${1}/noise0_scale2.0x_model.t7 ]; then
	th tools/export_model.lua -i models/${1}/noise0_scale2.0x_model.t7 -o models/${1}/noise0_scale2.0x_model.json
    fi
    if [ -f models/${1}/noise1_scale2.0x_model.t7 ] && [ ! -h models/${1}/noise1_scale2.0x_model.t7 ]; then
	th tools/export_model.lua -i models/${1}/noise1_scale2.0x_model.t7 -o models/${1}/noise1_scale2.0x_model.json
    fi
    if [ -f models/${1}/noise2_scale2.0x_model.t7 ] && [ ! -h models/${1}/noise2_scale2.0x_model.t7 ]; then
	th tools/export_model.lua -i models/${1}/noise2_scale2.0x_model.t7 -o models/${1}/noise2_scale2.0x_model.json
    fi
    if [ -f models/${1}/noise3_scale2.0x_model.t7 ] && [ ! -h models/${1}/noise3_scale2.0x_model.t7 ]; then
	th tools/export_model.lua -i models/${1}/noise3_scale2.0x_model.t7 -o models/${1}/noise3_scale2.0x_model.json
    fi
}
export_model vgg_7/art
export_model upconv_7/art
export_model upconv_7l/art
export_model vgg_7/photo
export_model upconv_7/photo
export_model upconv_7l/photo
export_model cunet/art
