local function load_nn()
   require 'torch'
   require 'nn'
end
local function load_cunn()
   require 'cutorch'
   require 'cunn'
end
local function load_cudnn()
   require 'cudnn'
   cudnn.benchmark = true
end
if w2nn then
   return w2nn
else
   pcall(load_cunn)
   pcall(load_cudnn)
   w2nn = {}
   require 'LeakyReLU'
   require 'LeakyReLU_deprecated'
   require 'DepthExpand2x'
   require 'PSNRCriterion'
   require 'ClippedWeightedHuberCriterion'
   return w2nn
end
