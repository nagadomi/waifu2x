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
local function make_data_parallel_table(model, gpus)
   if cudnn then
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark
      local dpt = nn.DataParallelTable(1, true, true)
	 :add(model, gpus)
	 :threads(function()
	       require 'pl'
	       local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
	       package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
	       require 'torch'
	       require 'cunn'
	       require 'w2nn'
	       local cudnn = require 'cudnn'
	       cudnn.fastest, cudnn.benchmark = fastest, benchmark
		 end)
      dpt.gradInput = nil
      model = dpt:cuda()
   else
      local dpt = nn.DataParallelTable(1, true, true)
	    :add(model, gpus)
	 :threads(function()
	       require 'pl'
	       local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
	       package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
	       require 'torch'
	       require 'cunn'
	       require 'w2nn'
		 end)
      dpt.gradInput = nil
      model = dpt:cuda()
   end
   return model
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

   function w2nn.load_model(model_path, force_cudnn, mode)
      mode = mode or "ascii"
      local model = torch.load(model_path, mode)
      if force_cudnn then
	 model = cudnn.convert(model, cudnn)
      end
      model:cuda():evaluate()
      return model
   end
   function w2nn.data_parallel(model, gpus)
      if #gpus > 1 then
	 return make_data_parallel_table(model, gpus)
      else
	 return model
      end
   end
   require 'LeakyReLU'
   require 'ClippedWeightedHuberCriterion'
   require 'ClippedMSECriterion'
   require 'SSIMCriterion'
   require 'InplaceClip01'
   require 'L1Criterion'
   require 'ShakeShakeTable'
   require 'PrintTable'
   require 'Print'
   require 'AuxiliaryLossTable'
   require 'AuxiliaryLossCriterion'
   require 'GradWeight'
   require 'RandomBinaryConvolution'
   require 'LBPCriterion'
   require 'EdgeFilter'
   require 'ScaleTable'
   return w2nn
end

