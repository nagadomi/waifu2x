# models/vgg_7/art vs models/upconv_7/art

Warning: This benchmark results is outdated. I will update soon.

## dataset

noise free 84 anime style arts. PNG format only.

## 2x

### filter=Sinc

| model         |  PSNR      | time   |
|---------------|------------|--------|
| vgg\_7/art    | 36.471     | 22.21  |
| upconv\_7/art | 37.087     |  9.44  |

### filter=Box
| model         |  PSNR      | time   |
|---------------|------------|--------|
| vgg\_7/art    | 36.456     | 22.05  |
| upconv\_7/art | 36.952     |  9.27  |

## 2x + noise reduction level 1, jpeg quality=80

### filter=Sinc
| model         |  PSNR      | time   |
|---------------|------------|--------|
| vgg\_7/art    | 32.989     | 27.79  |
| upconv\_7/art | 33.211     |  9.41  |

### filter=Box

| model         |  PSNR      | time   |
|---------------|------------|--------|
| vgg\_7/art    | 33.200     | 27.07  |
| upconv\_7/art | 33.322     |  9.33  |

### 2x + noise reduction level 2, jpeg quality=50,45

### filter=Sinc

| model         |  PSNR      | time   |
|---------------|------------|--------|
| vgg\_7/art    | 30.446     | 27.62  |
| upconv\_7/art | 30.658     |  9.45  |

### filter=Box

| model         |  PSNR      | time   |
|---------------|------------|--------|
| vgg\_7/art    | 30.513     | 27.37  |
| upconv\_7/art | 30.699     |  9.43  |
