#!/bin/sh

th convert_data.lua -data_dir ./data/ukbench

#th train.lua -style photo -method noise -noise_level 2 -data_dir ./data/ukbench -model_dir models/ukbench -test images/lena.png -nr_rate 0.9 -jpeg_sampling_factors 420 # -thread 4 -backend cudnn 
#th tools/cleanup_model.lua -model models/ukbench/noise2_model.t7 -oformat ascii

th train.lua -method scale -data_dir ./data/ukbench -model_dir models/ukbench -test images/lena.jpg # -thread 4 -backend cudnn
th tools/cleanup_model.lua -model models/ukbench/scale2.0x_model.t7 -oformat ascii
