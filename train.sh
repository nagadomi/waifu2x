#!/bin/sh

th convert_data.lua

th train.lua -method scale -model_dir models/anime_style_art_rgb -test images/miku_small.png -thread 4
th train.lua -method noise -noise_level 1 -style art -model_dir models/anime_style_art_rgb -test images/miku_noisy.png -thread 4
th train.lua -method noise -noise_level 2 -style art -model_dir models/anime_style_art_rgb -test images/miku_noisy.png -thread 4
