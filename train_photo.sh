#!/bin/sh

th convert_data.lua -style photo -data_dir ./data/photo -model_dir models/photo

th train.lua -style photo -method scale -data_dir ./data/photo -model_dir models/photo -test work/scale_test_photo.png -color rgb -random_unsharp_mask_rate 0.1 #-thread 4 -backend cudnn 
th train.lua -style photo -method noise -noise_level 1 -data_dir ./data/photo -model_dir models/photo -test work/noise_test_photo.jpg -color rgb -random_unsharp_mask_rate 0.5 -nr_rate 0.6
th train.lua -style photo -method noise -noise_level 2 -data_dir ./data/photo -model_dir models/photo -test work/noise_test_photo.jpg -color rgb -random_unsharp_mask_rate 0.5 -nr_rate 0.8
th train.lua -style photo -method noise -noise_level 3 -data_dir ./data/photo -model_dir models/photo -test work/noise_test_photo.jpg -color rgb -random_unsharp_mask_rate 0.5 -nr_rate 1
