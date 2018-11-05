#!/bin/sh

# convert data 
th convert_data.lua -max_training_image_size 1600

# scale
# after train, I selected `scale2.0x_model.86-1.t7`, because seemingly this model is best mesh-artifact free model. Maybe visual testing is required.
# ref: https://github.com/nagadomi/waifu2x/issues/125
th train.lua -save_history 1 -scale 2 -model upconv_7 -method scale -model_dir models/test/upconv_7_rev5 -downsampling_filters "Box,Sinc" -test query/pixel-art-small.png -backend cudnn -thread 4 -oracle_rate 0.0 -inner_epoch 2 -epoch 100

# noise_scale

th train.lua -save_history 1 -model upconv_7 -method noise_scale -noise_level 0 -model_dir models/test/upconv_7_rev6 -downsampling_filters "Box,Sinc" -test query/noise_test.jpg -backend cudnn -thread 4  -resume models/test/upconv_7_rev5/scale2.0x_model.t7 -style art
th train.lua -save_history 1 -model upconv_7 -method noise_scale -noise_level 1 -model_dir models/test/upconv_7_rev6 -downsampling_filters "Box,Sinc" -test query/noise_test.jpg -backend cudnn -thread 4  -resume models/test/upconv_7_rev5/scale2.0x_model.t7 -style art
th train.lua -save_history 1 -model upconv_7 -method noise_scale -noise_level 2 -model_dir models/test/upconv_7_rev6 -downsampling_filters "Box,Sinc" -test query/noise_test.jpg -backend cudnn -thread 4  -resume models/test/upconv_7_rev5/scale2.0x_model.t7 -style art
th train.lua -save_history 1 -model upconv_7 -method noise_scale -noise_level 3 -model_dir models/test/upconv_7_rev6 -downsampling_filters "Box,Sinc" -test query/noise_test.jpg -backend cudnn -thread 4  -resume models/test/upconv_7_rev5/scale2.0x_model.t7 -style art -nr_rate 1

# noise
th train.lua -save_history 1 -model vgg_7 -method noise -noise_level 0 -model_dir models/test/yuv420_rev2 -test query/noise_test.jpg -backend cudnn -thread 4 -style art -crop_size 32
th train.lua -save_history 1 -model vgg_7 -method noise -noise_level 1 -model_dir models/test/yuv420_rev2 -test query/noise_test.jpg -backend cudnn -thread 3 -style art -crop_size 32
th train.lua -save_history 1 -model vgg_7 -method noise -noise_level 2 -model_dir models/test/yuv420_rev2 -test query/noise_test.jpg -backend cudnn -thread 3 -style art -crop_size 32
th train.lua -save_history 1 -model vgg_7 -method noise -noise_level 3 -model_dir models/test/yuv420_rev2 -test query/noise_test.jpg -backend cudnn -thread 3 -style art -crop_size 32 -nr_rate 1 
