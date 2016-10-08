# Benchmarks

## Photo

Note: waifu2x's photo models was trained on [kou's photo collection](http://photosku.com/photo/category/%E6%92%AE%E5%BD%B1%E8%80%85/kou/).
Note: PSNR in this benchmark uses a MATLAB's rgb2ycbcr compatible function(dynamic range [16 235], not [0, 255]) for converting grayscale image. I think it's not correct PSNR. But many paper used this metric.

command: 
`th tools/benchmark.lua -dir <dataset_dir> -model1_dir <model_dir> -method scale -filter Catrom -color y -range_bug 1 -tta <0|1> -force_cudnn 1`

### Datasets

BSD100: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/ (100 test images in BSD300)
Urban100: https://github.com/jbhuang0604/SelfExSR

### 2x - PSNR 

| Dataset/Model | Bicubic       | vgg\_7/photo  | upconv\_7/photo  | upconv\_7l/photo | 
|---------------|---------------|---------------|------------------|------------------|
| BSD100        | 29.558        | 31.427        | 31.640           | 31.749           |
| Urban100      | 26.852        | 30.057        | 30.477           | 30.759           |

### 2x with TTA - PSNR 

Note: TTA is an ensemble technique that is supported by waifu2x. This method is 8x slower than non TTA method but it improves PSNR (~+0.1 on photo, ~+0.4 on art).

| Dataset/Model | Bicubic       | vgg\_7/photo  | upconv\_7/photo  | upconv\_7l/photo | 
|---------------|---------------|---------------|------------------|------------------|
| BSD100        | 29.558        | 31.474        | 31.705           | 31.812           |
| Urban100      | 26.852        | 30.140        | 30.599           | 30.868           |

### 2x - benchmark elapsed time (sec)

| Dataset/Model | vgg\_7/photo  | upconv\_7/photo  | upconv\_7l/photo | 
|---------------|---------------|------------------|------------------|
| BSD100        | 4.057         | 2.509            | 4.947            |
| Urban100      | 16.349        | 7.083            | 14.178           |

### 2x with TTA - benchmark elapsed time (sec)

| Dataset/Model | vgg\_7/photo  | upconv\_7/photo  | upconv\_7l/photo | 
|---------------|---------------|------------------|------------------|
| BSD100        | 36.611        | 20.219           | 42.486           |
| Urban100      | 132.416       | 65.125           | 129.916          |

## Art

command: 
`th tools/benchmark.lua -dir <dataset_dir> -model1_dir <model_dir> -method scale -filter Lanczos -color y -range_bug 1 -tta <0|1> -force_cudnn 1`

### Dataset

art_test: This dataset contains 85 various fan-arts. Sorry, This dataset is private. 

### 2x - PSNR 

| Dataset/Model | Bicubic       | vgg\_7/art  | upconv\_7/art  | upconv\_7l/art | 
|---------------|---------------|-------------|----------------|----------------|
| art_test      | 31.022        | 37.495      | 38.330         | 39.140         |

### 2x with TTA - PSNR 

| Dataset/Model | Bicubic       | vgg\_7/art  | upconv\_7/art  | upconv\_7l/art | 
|---------------|---------------|-------------|----------------|----------------|
| art_test      | 31.022        | 37.777      | 38.677         | 39.510         |

### 2x - benchmark elapsed time (sec)

| Dataset/Model | vgg\_7/art  | upconv\_7/art  | upconv\_7l/art | 
|---------------|-------------|----------------|----------------|
| art_test      | 20.681      | 7.683          | 17.667         |

### 2x with TTA - benchmark elapsed time (sec)

| Dataset/Model | vgg\_7/art  | upconv\_7/art  | upconv\_7l/art | 
|---------------|-------------|----------------|----------------|
| art_test      | 174.674     | 77.716         | 163.932        |

