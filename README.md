# waifu2x

Image Super-Resolution for anime/fan-art using Deep Convolutional Neural Networks.

Demo-Application can be found at http://waifu2x.udp.jp/ .

## Summary

Click to see the slide show.

![slide](https://raw.githubusercontent.com/nagadomi/waifu2x/master/images/slide.png)

## References

waifu2x is inspired by SRCNN [1]. 2D character picture (HatsuneMiku) is licensed under CC BY-NC by piapro [2].

- [1] Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, "Image Super-Resolution Using Deep Convolutional Networks", http://arxiv.org/abs/1501.00092
- [2] "For Creators", http://piapro.net/en_for_creators.html

## Public AMI
(maintenance)

## Dependencies

### Hardware
- NVIDIA GPU (Compute Capability 3.0 or later)

### Platform
- [Torch7](http://torch.ch/)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)
- [NVIDIA cuDNN](https://developer.nvidia.com/cuDNN)

### Packages (luarocks)
- cutorch
- cunn
- [cudnn](https://github.com/soumith/cudnn.torch)
- [graphicsmagick](https://github.com/clementfarabet/graphicsmagick)
- [turbo](https://github.com/kernelsauce/turbo)
- md5
- uuid

NOTE: Turbo 1.1.3 has bug in file uploading. Please install from the master branch on github.

## Installation

### Setting Up the Command Line Tool Environment
 (on Ubuntu 14.04)
 
#### Install Torch7

```
sudo apt-get install curl
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | sudo bash 
```
see [Torch (easy) install](https://github.com/torch/ezinstall)

#### Install CUDA and cuDNN.

Google! Search keyword is "install cuda ubuntu" and "install cudnn ubuntu"

#### Install packages

```
sudo luarocks install cutorch
sudo luarocks install cunn
sudo luarocks install cudnn
sudo apt-get install graphicsmagick libgraphicsmagick-dev
sudo luarocks install graphicsmagick
```

Test the waifu2x command line tool. 
```
th waifu2x.lua
```

### Setting Up the Web Application Environment (if you needed)

#### Install luajit 2.0.4

```
curl -O http://luajit.org/download/LuaJIT-2.0.4.tar.gz
tar -xzvf LuaJIT-2.0.4.tar.gz
cd LuaJIT-2.0.4
make
sudo make install
```

#### Install packages

Install luarocks packages.
```
sudo luarocks install md5
sudo luarocks install uuid
```

Install turbo.
```
git clone https://github.com/kernelsauce/turbo.git
cd turbo
sudo luarocks make rockspecs/turbo-dev-1.rockspec 
```

## Web Application

Please edit the first line in `web.lua`.
```
local ROOT = '/path/to/waifu2x/dir'
```
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
(You should use PNG! In my case, waifu2x is trained with 3000 high-resolution-beautiful-PNG images.)

Converting training data.
```
th convert_data.lua
```

### Training a Noise Reduction(level1) model

```
th train.lua -method noise -noise_level 1 -test images/miku_noisy.png
th cleanup_model.lua -model models/noise1_model.t7 -oformat ascii
```
You can check the performance of model with `models/noise1_best.png`.

### Training a Noise Reduction(level2) model

```
th train.lua -method noise -noise_level 2 -test images/miku_noisy.png
th cleanup_model.lua -model models/noise2_model.t7 -oformat ascii
```
You can check the performance of model with `models/noise2_best.png`.

### Training a 2x UpScaling model

```
th train.lua -method scale -scale 2 -test images/miku_small.png
th cleanup_model.lua -model models/scale2.0x_model.t7 -oformat ascii
```
You can check the performance of model with `models/scale2.0x_best.png`.
