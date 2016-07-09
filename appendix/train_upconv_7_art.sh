#!/bin/sh

# convert data 
th convert_data.lua -max_training_image_size 1600

# scale
th train.lua -save_history 1 -inner_epoch 1 -epoch 100 -scale 2 -model upconv_7 -method scale -model_dir models/test/upconv_7_rev3 -downsampling_filters "Box,Sinc" -test query/scale_test.png -backend cudnn -thread 4 

# noise_scale
th train.lua -save_history 1 -model upconv_7 -method noise_scale -noise_level 0 -model_dir models/test/yuv420_rev2 -downsampling_filters "Box,Sinc" -test query/noise_test.jpg -backend cudnn -thread 4  -resume models/test/upconv_7_rev3/scale2.0x_model.t7 -style art 
th train.lua -save_history 1 -model upconv_7 -method noise_scale -noise_level 1 -model_dir models/test/yuv420_rev2 -downsampling_filters "Box,Sinc" -test query/noise_test.jpg -backend cudnn -thread 4  -resume models/test/upconv_7_rev3/scale2.0x_model.t7 -style art 
th train.lua -save_history 1 -model upconv_7 -method noise_scale -noise_level 2 -model_dir models/test/yuv420_rev2 -downsampling_filters "Box,Sinc" -test query/noise_test.jpg -backend cudnn -thread 4  -resume models/test/upconv_7_rev3/scale2.0x_model.t7 -style art 
th train.lua -save_history 1 -model upconv_7 -method noise_scale -noise_level 3 -model_dir models/test/yuv420_rev2 -downsampling_filters "Box,Sinc" -test query/noise_test.jpg -backend cudnn -thread 4  -resume models/test/upconv_7_rev3/scale2.0x_model.t7 -style art -nr_rate 1

# noise
th train.lua -save_history 1 -model vgg_7 -method noise -noise_level 0 -model_dir models/test/yuv420_rev2 -test query/noise_test.jpg -backend cudnn -thread 4 -style art -crop_size 32
th train.lua -save_history 1 -model vgg_7 -method noise -noise_level 1 -model_dir models/test/yuv420_rev2 -test query/noise_test.jpg -backend cudnn -thread 3 -style art -crop_size 32
th train.lua -save_history 1 -model vgg_7 -method noise -noise_level 2 -model_dir models/test/yuv420_rev2 -test query/noise_test.jpg -backend cudnn -thread 3 -style art -crop_size 32
th train.lua -save_history 1 -model vgg_7 -method noise -noise_level 3 -model_dir models/test/yuv420_rev2 -test query/noise_test.jpg -backend cudnn -thread 3 -style art -crop_size 32 -nr_rate 1 
