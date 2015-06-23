#!/bin/sh

th waifu2x.lua -noise_level 1 -m noise_scale -i images/miku_small.png -o images/miku_small_waifu2x.png
th waifu2x.lua -noise_level 2 -m noise_scale -i images/miku_small_noisy.png -o images/miku_small_noisy_waifu2x.png
th waifu2x.lua -noise_level 2 -m noise -i images/miku_noisy.png -o images/miku_noisy_waifu2x.png
th waifu2x.lua -noise_level 2 -m noise_scale -i images/miku_CC_BY-NC_noisy.jpg -o images/miku_CC_BY-NC_noisy_waifu2x.png
th waifu2x.lua -noise_level 2 -m noise -i images/lena.png -o images/lena_waifu2x.png
th waifu2x.lua -m scale -model_dir models/ukbench -i images/lena.png -o images/lena_waifu2x_ukbench.png
