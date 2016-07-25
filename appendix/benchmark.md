# Benchmark results

Warning: This benchmark results is outdated. I will update soon.

## Usage

```
th tools/benchmark.lua -dir path/to/dataset_dir -method scale -color y -model1_dir path/to/model_dir
```

## Dataset

    photo_test: 300 various photos.
    art_test  : 90 artworks (PNG only).

## 2x upscaling model

| Dataset/Model | anime\_style\_art(Y) | anime\_style\_art\_rgb | photo   | ukbench|
|---------------|----------------------|------------------------|---------|--------|
| photo\_test   |                29.83 |                  29.81 |**29.89**|  29.86 |
| art\_test     |                36.02 |               **36.24**|  34.92  |  34.85 |

The evaluation metric is PSNR(Y only), higher is better.

## Denosing level 1 model

| Dataset/Model            | anime\_style\_art | anime\_style\_art\_rgb | photo   |
|--------------------------|-------------------|------------------------|---------|
| photo\_test Quality 80   |             36.07 |               **36.20**|   36.01 |
| photo\_test Quality 50,45|             31.72 |                 32.01  |**32.31**|
| art\_test Quality 80     |             40.39 |               **42.48**|   40.35 |
| art\_test Quality 50,45  |             35.45 |               **36.70**|   36.27 |

The evaluation metric is PSNR(RGB), higher is better.

## Denosing level 2 model

| Dataset/Model            | anime\_style\_art | anime\_style\_art\_rgb | photo   |
|--------------------------|-------------------|------------------------|---------|
| photo\_test Quality 80   |             34.03 |                  34.42 |**36.06**|
| photo\_test Quality 50,45|             31.95 |                  32.31 |**32.42**|
| art\_test Quality 80     |             39.20 |               **41.12**|   40.48 |
| art\_test Quality 50,45  |             36.14 |               **37.78**|   36.55 |

The evaluation metric is PSNR(RGB), higher is better.
