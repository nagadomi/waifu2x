#!/bin/sh

th train.lua -category photo -color rgb -color_noise 0 -overlay 0 -random_half 0 -epoch 300 -batch_size 1 -method noise -noise_level 1 -data_dir ukbench -model_dir models/ukbench2 -test photo2.jpg
th cleanup_model.lua -model models/ukbench2/noise1_model.t7 -oformat ascii

th train.lua -core 1 -category photo -color rgb -color_noise 0 -overlay 0 -random_half 0 -epoch 300 -batch_size 1 -method noise -noise_level 2 -data_dir ukbench -model_dir models/ukbench2 -test photo2.jpg
th cleanup_model.lua -model models/ukbench2/noise2_model.t7 -oformat ascii

th train.lua -category photo -color rgb -random_half 0 -epoch 400 -batch_size 1 -method scale -scale 2 -model_dir models/ukbench2 -data_dir ukbench -test photo2-noise.png
th cleanup_model.lua -model models/ukbench2/scale2.0x_model.t7 -oformat ascii

