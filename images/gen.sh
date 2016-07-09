#!/bin/sh

th waifu2x.lua -m scale -i images/miku_small.png -o images/miku_small_waifu2x.png
th waifu2x.lua -m noise_scale -noise_level 1 -i images/miku_small_noisy.png -o images/miku_small_noisy_waifu2x.png
th waifu2x.lua -m noise_scale -noise_level 1 -i images/miku_CC_BY-NC_noisy.jpg -o images/miku_CC_BY-NC_noisy_waifu2x.png
th waifu2x.lua -m noise -noise_level 2 -i images/miku_noisy.png -o images/miku_noisy_waifu2x.png
th waifu2x.lua -m noise -noise_level 2 -i images/lena.png -o images/lena_waifu2x_art_noise2.png

th waifu2x.lua -m scale -i images/tone.png -o images/tone_waifu2x.png

th waifu2x.lua -m scale -model_dir models/upconv_7/photo -i images/lena.png -o images/lena_waifu2x_photo.png
th waifu2x.lua -m noise_scale -noise_level 3 -model_dir models/upconv_7/photo -i images/lena.png -o images/lena_waifu2x_photo_noise3.png
th waifu2x.lua -m scale -model_dir models/upconv_7/photo -i images/city.jpg -o images/city_waifu2x_photo.png
