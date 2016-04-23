require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
require 'os'
require 'w2nn'
local srcnn = require 'srcnn'

local function rebuild(old_model, model)
   local new_model = srcnn.create(model, srcnn.backend(old_model), srcnn.color(old_model))
   local weight_from = old_model:findModules("nn.SpatialConvolutionMM")
   local weight_to = new_model:findModules("nn.SpatialConvolutionMM")

   assert(#weight_from == #weight_to)
   
   for i = 1, #weight_from do
      local from = weight_from[i]
      local to = weight_to[i]
      
      to.weight:copy(from.weight)
      to.bias:copy(from.bias)
   end
   new_model:cuda()
   new_model:evaluate()
   return new_model
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x rebuild cunn model")
cmd:text("Options:")
cmd:option("-i", "", 'Specify the input model')
cmd:option("-o", "", 'Specify the output model')
cmd:option("-model", "vgg_7", 'Specify the model architecture (vgg_7|vgg_12)')
cmd:option("-iformat", "ascii", 'Specify the input format (ascii|binary)')
cmd:option("-oformat", "ascii", 'Specify the output format (ascii|binary)')

local opt = cmd:parse(arg)
if not path.isfile(opt.i) then
   cmd:help()
   os.exit(-1)
end
local old_model = torch.load(opt.i, opt.iformat)
local new_model = rebuild(old_model, opt.model)
torch.save(opt.o, new_model, opt.oformat)
