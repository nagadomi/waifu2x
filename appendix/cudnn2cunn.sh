#!/bin/bash

th tools/cudnn2cunn.lua -i models/test/test/upcunet_release/scale2.0x_model.t7 -o models/cunet/art/scale2.0x_model.t7
for i in 0 1 2 3
do
    th tools/cudnn2cunn.lua -i models/test/cunet_release/noise${i}_model.t7 -o models/cunet/art/noise${i}_model.t7
    th tools/cudnn2cunn.lua -i models/test/cunet_release/noise${i}_scale2.0x_model.t7 -o models/cunet/art/noise${i}_scale2.0x_model.t7
done
