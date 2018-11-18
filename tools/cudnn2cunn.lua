require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
require 'os'
require 'w2nn'
local srcnn = require 'srcnn'

local function cudnn2cunn(cudnn_model)
   local name = srcnn.name(cudnn_model)
   local cunn_model = srcnn[name]('cunn', srcnn.channels(cudnn_model))
   local param_layers = {
      {cunn="nn.SpatialConvolutionMM", cudnn="cudnn.SpatialConvolution", attr={"bias", "weight"}},
      {cunn="nn.SpatialDilatedConvolution", cudnn="cudnn.SpatialDilatedConvolution", attr={"bias", "weight"}},
      {cunn="nn.SpatialFullConvolution", cudnn="cudnn.SpatialFullConvolution", attr={"bias", "weight"}},
      {cunn="nn.Linear", cudnn="nn.Linear", attr={"bias", "weight"}}
   }
   for i = 1, #param_layers do
      local p = param_layers[i]
      local weight_from = cudnn_model:findModules(p.cudnn)
      local weight_to = cunn_model:findModules(p.cunn)
      print(p.cudnn, #weight_from)
      assert(#weight_from == #weight_to)
   
      for i = 1, #weight_from do
	 local from = weight_from[i]
	 local to = weight_to[i]
	 to.weight:copy(from.weight)
	 if to.bias then
	    to.bias:copy(from.bias)
	 end
      end
   end
   cunn_model:cuda()
   cunn_model:evaluate()
   return cunn_model
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x cudnn model to cunn model converter")
cmd:text("Options:")
cmd:option("-i", "", 'Specify the input cunn model')
cmd:option("-o", "", 'Specify the output cudnn model')
cmd:option("-iformat", "ascii", 'Specify the input format (ascii|binary)')
cmd:option("-oformat", "ascii", 'Specify the output format (ascii|binary)')

local opt = cmd:parse(arg)
if not path.isfile(opt.i) then
   cmd:help()
   os.exit(-1)
end
local cudnn_model = torch.load(opt.i, opt.iformat)
local cunn_model = cudnn2cunn(cudnn_model)
torch.save(opt.o, cunn_model, opt.oformat)
