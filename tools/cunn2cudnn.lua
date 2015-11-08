require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
require 'os'
require 'w2nn'
local srcnn = require 'srcnn'

local function cunn2cudnn(cunn_model)
   local cudnn_model = srcnn.waifu2x_cudnn(srcnn.channels(cunn_model))
   local weight_from = cunn_model:findModules("nn.SpatialConvolutionMM")
   local weight_to = cudnn_model:findModules("cudnn.SpatialConvolution")

   assert(#weight_from == #weight_to)
   
   for i = 1, #weight_from do
      local from = weight_from[i]
      local to = weight_to[i]
      
      to.weight:copy(from.weight)
      to.bias:copy(from.bias)
   end
   cudnn_model:cuda()
   cudnn_model:evaluate()
   return cudnn_model
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x cunn model to cudnn model converter")
cmd:text("Options:")
cmd:option("-i", "", 'Specify the input cudnn model')
cmd:option("-o", "", 'Specify the output cunn model')
cmd:option("-iformat", "ascii", 'Specify the input format (ascii|binary)')
cmd:option("-oformat", "ascii", 'Specify the output format (ascii|binary)')

local opt = cmd:parse(arg)
if not path.isfile(opt.i) then
   cmd:help()
   os.exit(-1)
end
local cunn_model = torch.load(opt.i, opt.iformat)
local cudnn_model = cunn2cudnn(cunn_model)
torch.save(opt.o, cudnn_model, opt.oformat)
