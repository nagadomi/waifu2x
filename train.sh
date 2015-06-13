#!/bin/sh

th train.lua -method noise -noise_level 1 -model_dir models/anime_style_art -test images/miku_noisy.png
th cleanup_model.lua -model models/anime_style_art/noise1_model.t7 -oformat ascii

th train.lua -method noise -noise_level 2 -model_dir models/anime_style_art -test images/miku_noisy.png
th cleanup_model.lua -model models/anime_style_art/noise2_model.t7 -oformat ascii

th train.lua -method scale -scale 2 -model_dir models/anime_style_art -test images/miku_small.png
th cleanup_model.lua -model models/anime_style_art/scale2.0x_model.t7 -oformat ascii
