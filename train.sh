#!/bin/sh

th convert_data.lua

th train.lua -method scale -model_dir models/anime_style_art_rgb -test images/miku_small.png # -thread 4 -backend cudnn
th train.lua -method noise -noise_level 1 -style art -model_dir models/anime_style_art_rgb -test images/miku_noisy.png
th train.lua -method noise -noise_level 2 -style art -model_dir models/anime_style_art_rgb -test images/miku_noisy.png
th train.lua -method noise -noise_level 3 -style art -model_dir models/anime_style_art_rgb -test images/miku_noisy.png -nr_rate 1
