-- adapted from https://github.com/marcan/cl-waifu2x
require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
require 'w2nn'
local cjson = require "cjson"

function export(model, output)
   local jmodules = {}
   local modules = model:findModules("nn.SpatialConvolutionMM")
   if #modules == 0 then
      -- cudnn model
      modules = model:findModules("cudnn.SpatialConvolution")
   end
   for i = 1, #modules, 1 do
      local module = modules[i]
      local jmod = {
	 kW = module.kW,
	 kH = module.kH,
	 nInputPlane = module.nInputPlane,
	 nOutputPlane = module.nOutputPlane,
	 bias = torch.totable(module.bias:float()),
	 weight = torch.totable(module.weight:float():reshape(module.nOutputPlane, module.nInputPlane, module.kW, module.kH))
      }
      table.insert(jmodules, jmod)
   end
   local fp = io.open(output, "w")
   if not fp then
      error("IO Error: " .. output)
   end
   fp:write(cjson.encode(jmodules))
   fp:close()
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x export model")
cmd:text("Options:")
cmd:option("-i", "input.t7", 'Specify the input torch model')
cmd:option("-o", "output.json", 'Specify the output json file')
cmd:option("-iformat", "ascii", 'Specify the input format (ascii|binary)')

local opt = cmd:parse(arg)
if not path.isfile(opt.i) then
   cmd:help()
   os.exit(-1)
end
local model = torch.load(opt.i, opt.iformat)
export(model, opt.o)
