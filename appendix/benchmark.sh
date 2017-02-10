#!/bin/sh
set -x

benchmark_photo() {
    dir=./benchmarks/${1}/${2}/${3}
    mkdir -p ${dir}
    th tools/benchmark.lua -dir data/${1} -model1\_dir models/${2}/photo -method scale -filter Catrom -color y -range\_bug 1 -tta ${3} -force_cudnn 1 -output_dir ${dir} -save_info 1 -show_progress 0 
}
run_benchmark_photo() {
    for tta in 0 1
    do
	for dataset in bsd100 urban100
	do
	    benchmark_photo ${dataset} vgg_7 ${tta}
	    benchmark_photo ${dataset} upconv_7 ${tta}
	    benchmark_photo ${dataset} upconv_7l ${tta}
	done
    done
}
benchmark_art() {
    dir=./benchmarks/${1}/${2}/${3}
    mkdir -p ${dir}
    th tools/benchmark.lua -dir data/${1} -model1\_dir models/${2}/art -method scale -filter Lanczos -color y -range\_bug 1 -tta ${3} -force_cudnn 1 -output_dir ${dir} -save_info 1 -show_progress 0 
}
run_benchmark_art() {
    for tta in 0 1
    do
	benchmark_art art_test vgg_7 ${tta}
	benchmark_art art_test upconv_7 ${tta}
	benchmark_art art_test upconv_7l ${tta}
    done
}
#run_benchmark_photo
run_benchmark_art 
