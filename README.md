# waifu2x

Image Super-Resolution for anime-style-art using Deep Convolutional Neural Networks.

Demo-Application can be found at http://waifu2x.udp.jp/ .

## Summary

Click to see the slide show.

![slide](https://raw.githubusercontent.com/nagadomi/waifu2x/master/images/slide.png)

## References

waifu2x is inspired by SRCNN [1]. 2D character picture (HatsuneMiku) is licensed under CC BY-NC by piapro [2].

- [1] Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, "Image Super-Resolution Using Deep Convolutional Networks", http://arxiv.org/abs/1501.00092
- [2] "For Creators", http://piapro.net/en_for_creators.html

## Public AMI
```
AMI ID: ami-0be01e4f
AMI NAME: waifu2x-server
Instance Type: g2.2xlarge
Region: US West (N.California)
OS: Ubuntu 14.04
User: ubuntu
Created at: 2015-08-12
```

## Third Party Software
[Third-Party](https://github.com/nagadomi/waifu2x/wiki/Third-Party)

## Dependencies

### Hardware
- NVIDIA GPU

### Platform
- [Torch7](http://torch.ch/)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)

### lualocks packages (excludes torch7's default packages)
- md5
- uuid
- [turbo](https://github.com/kernelsauce/turbo)

## Installation

### Setting Up the Command Line Tool Environment
 (on Ubuntu 14.04)

#### Install CUDA

See: [NVIDIA CUDA Getting Started Guide for Linux](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#ubuntu-installation)

Download [CUDA](http://developer.nvidia.com/cuda-downloads)

```
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

#### Install Torch7

See: [Getting started with Torch](http://torch.ch/docs/getting-started.html)

#### Validation

Test the waifu2x command line tool.
```
th waifu2x.lua
```

### Setting Up the Web Application Environment (if you needed)

#### Install packages

```
luarocks install md5
luarocks install uuid
PREFIX=$HOME/torch/install luarocks install turbo
```

## Web Application
Run.
```
th web.lua
```

View at: http://localhost:8812/

## Command line tools

### Noise Reduction
```
th waifu2x.lua -m noise -noise_level 1 -i input_image.png -o output_image.png
```
```
th waifu2x.lua -m noise -noise_level 2 -i input_image.png -o output_image.png
```

### 2x Upscaling
```
th waifu2x.lua -m scale -i input_image.png -o output_image.png
```

### Noise Reduction + 2x Upscaling
```
th waifu2x.lua -m noise_scale -noise_level 1 -i input_image.png -o output_image.png
```
```
th waifu2x.lua -m noise_scale -noise_level 2 -i input_image.png -o output_image.png
```

See also `images/gen.sh`.

### Video Encoding

\* `avconv` is `ffmpeg` on Ubuntu 14.04.

Extracting images and audio from a video. (range: 00:09:00 ~ 00:12:00)
```
mkdir frames
avconv -i data/raw.avi -ss 00:09:00 -t 00:03:00 -r 24 -f image2 frames/%06d.png
avconv -i data/raw.avi -ss 00:09:00 -t 00:03:00 audio.mp3
```

Generating a image list.
```
find ./frames -name "*.png" |sort > data/frame.txt
```

waifu2x (for example, noise reduction)
```
mkdir new_frames
th waifu2x.lua -m noise -noise_level 1 -resume 1 -l data/frame.txt -o new_frames/%d.png
```

Generating a video from waifu2xed images and audio.
```
avconv -f image2 -r 24 -i new_frames/%d.png -i audio.mp3 -r 24 -vcodec libx264 -crf 16 video.mp4
```

## Training Your Own Model

### Data Preparation

Genrating a file list.
```
find /path/to/image/dir -name "*.png" > data/image_list.txt
```
(You should use PNG! In my case, waifu2x is trained with 3000 high-resolution-noise-free-PNG images.)

Converting training data.
```
th convert_data.lua
```

### Training a Noise Reduction(level1) model

```
mkdir models/my_model
th train.lua -model_dir models/my_model -method noise -noise_level 1 -test images/miku_noisy.png
th cleanup_model.lua -model models/my_model/noise1_model.t7 -oformat ascii
# usage
th waifu2x.lua -model_dir models/my_model -m noise -noise_level 1 -i images/miku_noisy.png -o output.png
```
You can check the performance of model with `models/my_model/noise1_best.png`.

### Training a Noise Reduction(level2) model

```
th train.lua -model_dir models/my_model -method noise -noise_level 2 -test images/miku_noisy.png
th cleanup_model.lua -model models/my_model/noise2_model.t7 -oformat ascii
# usage
th waifu2x.lua -model_dir models/my_model -m noise -noise_level 2 -i images/miku_noisy.png -o output.png
```
You can check the performance of model with `models/my_model/noise2_best.png`.

### Training a 2x UpScaling model

```
th train.lua -model_dir models/my_model -method scale -scale 2 -test images/miku_small.png
th cleanup_model.lua -model models/my_model/scale2.0x_model.t7 -oformat ascii
# usage
th waifu2x.lua -model_dir models/my_model -m scale -scale 2 -i images/miku_small.png -o output.png
```
You can check the performance of model with `models/my_model/scale2.0x_best.png`.
