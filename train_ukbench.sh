#!/bin/sh

th convert_data.lua -data_dir ./data/ukbench

th train.lua -method scale -data_dir ./data/ukbench -model_dir models/ukbench -test images/lena.jpg -thread 4
th tools/cleanup_model.lua -model models/ukbench/scale2.0x_model.t7 -oformat ascii
