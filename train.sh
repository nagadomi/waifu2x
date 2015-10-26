#!/bin/sh

th convert_data.lua

th train.lua -color rgb -random_half 1 -jpeg_sampling_factors 444 -color_noise 0 -overlay 0 -epoch 200 -method noise -noise_level 1 -crop_size 46 -batch_size 8  -model_dir models/anime_style_art_rgb -test images/miku_noisy.jpg -validation_ratio 0.1 -active_cropping_rate 0.5 -active_cropping_tries 10 -validation_crops 80
th cleanup_model.lua -model models/anime_style_art_rgb/noise1_model.t7 -oformat ascii

th train.lua -color rgb -random_half 1 -jpeg_sampling_factors 444 -color_noise 0 -overlay 0 -epoch 200 -method noise -noise_level 2 -crop_size 46 -batch_size 8  -model_dir models/anime_style_art_rgb -test images/miku_noisy.jpg -validation_ratio 0.1 -active_cropping_rate 0.5 -active_cropping_tries 10 -validation_crops 80
th cleanup_model.lua -model models/anime_style_art_rgb/noise2_model.t7 -oformat ascii

th train.lua -color rgb -random_half 1 -jpeg_sampling_factors 444 -color_noise 0 -overlay 0 -epoch 200 -method scale -crop_size 46 -batch_size 8 -model_dir models/anime_style_art_rgb -test images/miku_small_noisy.jpg -active_cropping_rate 0.5 -active_cropping_tries 10 -validation_ratio 0.1 -validation_crops 80
th cleanup_model.lua -model models/anime_style_art_rgb/scale2.0x_model.t7 -oformat ascii

