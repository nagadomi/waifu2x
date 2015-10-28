-- adapted from https://github.com/marcan/cl-waifu2x
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
require 'w2nn'
local cjson = require "cjson"

local model = torch.load(arg[1], "ascii")

local jmodules = {}
local modules = model:findModules("nn.SpatialConvolutionMM")
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

io.write(cjson.encode(jmodules))
