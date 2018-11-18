#!/bin/sh

MODEL_DIR=models/test/cunet_release/
THREADS=16
mkdir -p ${MODEL_DIR}

# convert data
th convert_data.lua -max_training_image_size 1600

# scale
th train.lua -method scale -save_history 1 -model upcunet -downsampling_filters "Box,Box,Box,Box,Sinc,Sinc,Sinc,Sinc,Catrom" -model_dir ${MODEL_DIR} -test query/scale_test.png -thread ${THREADS} -backend cudnn -oracle_rate 0.0 -max_size 512 -loss aux_lbp -update_criterion loss -crop_size 104 -validation_crops 64 -inner_epoch 2 -epoch 200 -patches 32 -batch_size 8 > ${MODEL_DIR}/train.log 2>&1 

# noise scale 0
th train.lua -save_history 1 -model upcunet -method noise_scale -noise_level 0 -style art \
-downsampling_filters "Box,Box,Box,Box,Sinc,Sinc,Sinc,Sinc,Catrom" -model_dir ${MODEL_DIR} -test query/noise_test.jpg -thread ${THREADS} -backend cudnn -oracle_rate 0.1 -max_size 512 -loss aux_lbp -update_criterion loss -crop_size 104 -validation_crops 64 -inner_epoch 4 -epoch 30 -patches 32 -batch_size 8 \
-resume ${MODEL_DIR}/scale2.0x_model.t7 \
 > ${MODEL_DIR}/train_noise_scale0.log 2>&1 

# noise scale 1
th train.lua -save_history 1 -model upcunet -method noise_scale -noise_level 1 -style art \
-downsampling_filters "Box,Box,Box,Box,Sinc,Sinc,Sinc,Sinc,Catrom" -model_dir ${MODEL_DIR} -test query/noise_test.jpg -thread ${THREADS} -backend cudnn -oracle_rate 0.1 -max_size 512 -loss aux_lbp -update_criterion loss -crop_size 104 -validation_crops 64 -inner_epoch 4 -epoch 30 -patches 32 -batch_size 8 \
-resume ${MODEL_DIR}/scale2.0x_model.t7 \
 > ${MODEL_DIR}/train_noise_scale1.log 2>&1 

# noise scale 2
th train.lua -save_history 1 -model upcunet -method noise_scale -noise_level 2 -style art \
-downsampling_filters "Box,Box,Box,Box,Sinc,Sinc,Sinc,Sinc,Catrom" -model_dir ${MODEL_DIR} -test query/noise_test.jpg -thread ${THREADS} -backend cudnn -oracle_rate 0.1 -max_size 512 -loss aux_lbp -update_criterion loss -crop_size 104 -validation_crops 64 -inner_epoch 4 -epoch 30 -patches 32 -batch_size 8 \
-resume ${MODEL_DIR}/scale2.0x_model.t7 \
 > ${MODEL_DIR}/train_noise_scale2.log 2>&1 


# noise scale 3
th train.lua -save_history 1 -model upcunet -method noise_scale -noise_level 3 -style art \
-downsampling_filters "Box,Box,Box,Box,Sinc,Sinc,Sinc,Sinc,Catrom" -model_dir ${MODEL_DIR} -test query/noise_test.jpg -thread ${THREADS} -backend cudnn -oracle_rate 0.1 -max_size 512 -loss aux_lbp -update_criterion loss -crop_size 104 -validation_crops 64 -inner_epoch 4 -epoch 30 -patches 32 -batch_size 8 -nr_rate 1 \
-resume ${MODEL_DIR}/scale2.0x_model.t7 \
 > ${MODEL_DIR}/train_noise_scale3.log 2>&1 


# noise 0
th train.lua -save_history 1 -model cunet -method noise -noise_level 0 -model_dir ${MODEL_DIR} -test query/noise_test.jpg -backend cudnn -thread ${THREADS} -style art \
-crop_size 88 -validation_crops 64 -patches 16 -batch_size 8 -epoch 50 -max_size 512 \
-loss aux_lbp -update_criterion loss \
-oracle_rate 0.1 \
> ${MODEL_DIR}/train_noise0.log 2>&1 

# noise 1
th train.lua -save_history 1 -model cunet -method noise -noise_level 1 -model_dir ${MODEL_DIR} -test query/noise_test.jpg -backend cudnn -thread ${THREADS} -style art \
-crop_size 88 -validation_crops 64 -patches 16 -batch_size 8 -epoch 50 -max_size 512 \
-loss aux_lbp -update_criterion loss \
-oracle_rate 0.1 \
> ${MODEL_DIR}/train_noise1.log 2>&1 

# noise 2
th train.lua -save_history 1 -model cunet -method noise -noise_level 2 -model_dir ${MODEL_DIR} -test query/noise_test.jpg -backend cudnn -thread ${THREADS} -style art \
-crop_size 88 -validation_crops 64 -patches 16 -batch_size 8 -epoch 50 -max_size 512 \
-loss aux_lbp -update_criterion loss \
-oracle_rate 0.1 \
> ${MODEL_DIR}/train.log 2>&1 

# noise3

th train.lua -save_history 1 -model cunet -method noise -noise_level 3 -model_dir ${MODEL_DIR} -test query/noise_test.jpg -backend cudnn -thread ${THREADS} -style art -nr_rate 1 \
-crop_size 88 -validation_crops 64 -patches 16 -batch_size 8 -epoch 50 -max_size 512 \
-loss aux_lbp -update_criterion loss \
-oracle_rate 0.1 \
> ${MODEL_DIR}/train_noise3.log 2>&1 
