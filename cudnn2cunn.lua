require 'cunn'
require 'cudnn'
require 'cutorch'
require './lib/LeakyReLU'
local srcnn = require 'lib/srcnn'

local function cudnn2cunn(cudnn_model)
   local cunn_model = srcnn.waifu2x("y")
   local from_seq = cudnn_model:findModules("cudnn.SpatialConvolution")
   local to_seq = cunn_model:findModules("nn.SpatialConvolutionMM")

   for i = 1, #from_seq do
      local from = from_seq[i]
      local to = to_seq[i]
      to.weight:copy(from.weight)
      to.bias:copy(from.bias)
   end
   cunn_model:cuda()
   cunn_model:evaluate()
   return cunn_model
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text("convert cudnn model to cunn model ")
cmd:text("Options:")
cmd:option("-model", "./model.t7", 'path of cudnn model file')
cmd:option("-iformat", "ascii", 'input format')
cmd:option("-oformat", "ascii", 'output format')

local opt = cmd:parse(arg)
local cudnn_model = torch.load(opt.model, opt.iformat)
local cunn_model = cudnn2cunn(cudnn_model)
torch.save(opt.model, cunn_model, opt.oformat)
