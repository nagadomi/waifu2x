#!/bin/sh

# convert data
th convert_data.lua -style photo -data_dir ./data/photo -max_training_image_size 1600

# scale
th train.lua -save_history 1 -model upconv_7 -downsampling_filters "Box,Sinc,Catrom" -style photo -data_dir ./data/photo -model_dir models/test/new_photo_rev2 -test query/machine.png -color rgb -random_unsharp_mask_rate 0.3 -thread 3 -backend cudnn -oracle_rate 0.05

# noise_scale
th train.lua -save_history 1 -model upconv_7 -downsampling_filters "Box,Sinc,Catrom" -style photo -method noise_scale -noise_level 0 -data_dir ./data/photo -model_dir models/test/new_photo_rev2 -test query/bsd500.jpg -color rgb -random_unsharp_mask_rate 0.5 -thread 3 -backend cudnn -nr_rate 0.3 -resume models/test/new_photo_rev2/scale2.0x_model.t7 -oracle_rate 0.0 -active_cropping_rate 0 -resize_blur_min 1 -resize_blur_max 1
th train.lua -save_history 1 -model upconv_7 -downsampling_filters "Box,Sinc,Catrom" -style photo -method noise_scale -noise_level 1 -data_dir ./data/photo -model_dir models/test/new_photo_rev2 -test query/bsd500.jpg -color rgb -random_unsharp_mask_rate 0.5 -thread 3 -backend cudnn -nr_rate 0.3 -resume models/test/new_photo_rev2/scale2.0x_model.t7 -oracle_rate 0.0 -active_cropping_rate 0 -resize_blur_min 1 -resize_blur_max 1
th train.lua -save_history 1 -model upconv_7 -downsampling_filters "Box,Sinc,Catrom" -style photo -method noise_scale -noise_level 2 -data_dir ./data/photo -model_dir models/test/new_photo_rev2 -test query/bsd500.jpg -color rgb -random_unsharp_mask_rate 0.5 -thread 3 -backend cudnn -nr_rate 0.6 -resume models/test/new_photo_rev2/scale2.0x_model.t7 -oracle_rate 0.0 -active_cropping_rate 0 -resize_blur_min 1 -resize_blur_max 1
th train.lua -save_history 1 -model upconv_7 -downsampling_filters "Box,Sinc,Catrom" -style photo -method noise_scale -noise_level 3 -data_dir ./data/photo -model_dir models/test/new_photo_rev2 -test query/bsd500.jpg -color rgb -random_unsharp_mask_rate 0.5 -thread 3 -backend cudnn -nr_rate 1 -resume models/test/new_photo_rev2/scale2.0x_model.t7 -oracle_rate 0.0 -active_cropping_rate 0 -resize_blur_min 1 -resize_blur_max 1

# noise
th train.lua -save_history 1 -model vgg_7 -style photo -method noise -noise_level 0 -data_dir ./data/photo -model_dir models/test/new_photo -test query/bsd500.jpg -color rgb -random_unsharp_mask_rate 0.5 -thread 3 -backend cudnn -nr_rate 0.3 -oracle_rate 0.0 -active_cropping_rate 0 -resize_blur_min 1 -resize_blur_max 1 -crop_size 32 
th train.lua -save_history 1 -model vgg_7 -style photo -method noise -noise_level 1 -data_dir ./data/photo -model_dir models/test/new_photo -test query/bsd500.jpg -color rgb -random_unsharp_mask_rate 0.5 -thread 3 -backend cudnn -nr_rate 0.3 -oracle_rate 0.0 -active_cropping_rate 0 -resize_blur_min 1 -resize_blur_max 1 -crop_size 32 
th train.lua -save_history 1 -model vgg_7 -style photo -method noise -noise_level 2 -data_dir ./data/photo -model_dir models/test/new_photo -test query/bsd500.jpg -color rgb -random_unsharp_mask_rate 0.5 -thread 3 -backend cudnn -nr_rate 0.6 -oracle_rate 0.0 -active_cropping_rate 0 -resize_blur_min 1 -resize_blur_max 1 -crop_size 32 
th train.lua -save_history 1 -model vgg_7 -style photo -method noise -noise_level 3 -data_dir ./data/photo -model_dir models/test/new_photo -test query/bsd500.jpg -color rgb -random_unsharp_mask_rate 0.5 -thread 3 -backend cudnn -nr_rate 1 -oracle_rate 0.0 -active_cropping_rate 0 -resize_blur_min 1 -resize_blur_max 1 -crop_size 32 
