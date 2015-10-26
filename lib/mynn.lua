local function load_cunn()
   require 'nn'
   require 'cunn'
end
local function load_cudnn()
   require 'cudnn'
   cudnn.fastest = true
end
if mynn then
   return mynn
else
   load_cunn()
   --load_cudnn()
   mynn = {}
   require './LeakyReLU'
   require './LeakyReLU_deprecated'
   require './DepthExpand2x'
   require './RGBWeightedMSECriterion'
   return mynn
end
