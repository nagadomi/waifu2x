# Benchmarks

## Photo

Note: waifu2x's photo models was trained on the blending dataset of [kou's photo collection](http://photosku.com/photo/category/%E6%92%AE%E5%BD%B1%E8%80%85/kou/) and [ukbench](http://vis.uky.edu/~stewe/ukbench/).

Note: PSNR in this benchmark uses a [MATLAB's rgb2ycbcr](https://jp.mathworks.com/help/images/ref/rgb2ycbcr.html?lang=en) compatible function (dynamic range [16 235], not [0 255]) for converting grayscale image. I think it's not correct PSNR. But many paper used this metric.

command: 
`th tools/benchmark.lua -dir <dataset_dir> -model1_dir <model_dir> -method scale -filter Catrom -color y -range_bug 1 -tta <0|1> -force_cudnn 1`

### Datasets

BSD100: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/ (100 test images in BSDS300)
Urban100: https://github.com/jbhuang0604/SelfExSR

### 2x - PSNR 

| Dataset/Model | Bicubic       | vgg\_7/photo  | upconv\_7/photo  | upconv\_7l/photo | resnet_14l/photo | 
|---------------|---------------|---------------|------------------|------------------|--------------------|
| BSD100        | 29.558        | 31.427        | 31.640           | 31.749           | 31.847             |
| Urban100      | 26.852        | 30.057        | 30.477           | 30.759           | 31.016             |

### 2x - benchmark time (sec)

| Dataset/Model | vgg\_7/photo  | upconv\_7/photo  | upconv\_7l/photo | resnet_14l/photo |
|---------------|---------------|------------------|------------------|--------------------|
| BSD100        | 4.057         | 2.509            | 4.947            | 6.86               |
| Urban100      | 16.349        | 7.083            | 14.178           | 27.87              |

### 2x with TTA - PSNR 

Note: TTA is an ensemble technique that is supported by waifu2x. TTA method is 8x slower than non TTA method but it improves PSNR (~+0.1 on photo, ~+0.4 on art).

| Dataset/Model | Bicubic       | vgg\_7/photo  | upconv\_7/photo  | upconv\_7l/photo | resnet_14l/photo | 
|---------------|---------------|---------------|------------------|------------------|--------------------|
| BSD100        | 29.558        | 31.474        | 31.705           | 31.812           | 31.915             |
| Urban100      | 26.852        | 30.140        | 30.599           | 30.868           | 31.162             |

### 2x with TTA - benchmark time (sec)

| Dataset/Model | vgg\_7/photo  | upconv\_7/photo  | upconv\_7l/photo | resnet_14l/photo |
|---------------|---------------|------------------|------------------|--------------------|
| BSD100        | 36.611        | 20.219           | 42.486           | 60.38              |
| Urban100      | 132.416       | 65.125           | 129.916          | 255.20             |

## Art

command: See `appendix/benchmark.sh`

### Dataset

art_test: This dataset contains 84 various fan-arts. Sorry, This dataset is private.

### 2x - PSNR 

| Filter/Model   | Bicubic       | vgg\_7/art  | upconv\_7/art  | cunet/art      | 
|----------------|---------------|-------------|----------------|----------------|
| Lanczos        | 31.022        | 37.495      | 38.330         | 39.886         |
| Sinc           | 30.947        | 37.722      | 38.538         | 40.312         |
| Catrom(Bicubic)| 30.663        | 37.278      | 37.189         | 40.184         |
| Box            | 30.891        | 37.709      | 38.410         | 39.672         |

### 2x - benchmark time (sec)

| Dataset/Model | vgg\_7/art  | upconv\_7/art  | cunet/art | 
|---------------|-------------|----------------|----------------|
| art_test      | 24.153      | 10.794         | 24.222         |

### 2x with TTA - PSNR 

| Filter/Model   | Bicubic       | vgg\_7/art  | upconv\_7/art  | cunet/art      | 
|----------------|---------------|-------------|----------------|----------------|
| Lanczos        | 31.022        | 37.777      | 38.677         | 40.289         |
| Sinc           | 30.947        | 38.005      | 38.883         | 40.707         |
| Catrom(Bicubic)| 30.663        | 37.498      | 37.417         | 40.592         |
| Box            | 30.891        | 38.032      | 38.768         | 40.032         |

### 2x with TTA - benchmark time (sec)

| Dataset/Model | vgg\_7/art  | upconv\_7/art  | cunet/art       | 
|---------------|-------------|----------------|----------------|
| art_test      | 207.217     | 99.151         | 211.520        |
