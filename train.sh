#!/bin/sh

th train.lua -method noise -noise_level 1 -test images/miku_noisy.png
th cleanup_model.lua -model models/noise1_model.t7 -oformat ascii

th train.lua -method noise -noise_level 2 -test images/miku_noisy.png
th cleanup_model.lua -model models/noise2_model.t7 -oformat ascii

th train.lua -method scale -scale 2 -test images/miku_small.png
th cleanup_model.lua -model models/scale2.0x_model.t7 -oformat ascii
