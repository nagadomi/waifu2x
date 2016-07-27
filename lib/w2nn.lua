local function load_nn()
   require 'torch'
   require 'nn'
end
local function load_cunn()
   require 'cutorch'
   require 'cunn'
end
local function load_cudnn()
   cudnn = require('cudnn')
end
if w2nn then
   return w2nn
else
   w2nn = {}
   local state, ret = pcall(load_cunn)
   if not state then
      error("Failed to load CUDA modules. Please check the CUDA Settings.\n---\n" .. ret)
   end
   pcall(load_cudnn)

   function w2nn.load_model(model_path, force_cudnn)
      local model = torch.load(model_path, "ascii")
      if force_cudnn then
	 model = cudnn.convert(model, cudnn)
      end
      model:cuda():evaluate()
      return model
   end
   require 'LeakyReLU'
   require 'ClippedWeightedHuberCriterion'
   require 'ClippedMSECriterion'
   return w2nn
end
