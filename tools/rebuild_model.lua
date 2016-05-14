require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
require 'os'
require 'w2nn'
local srcnn = require 'srcnn'

local function rebuild(old_model, model, backend)
   local targets = {
      {"nn.SpatialConvolutionMM", 
       {cunn = "nn.SpatialConvolutionMM", 
	cudnn = "cudnn.SpatialConvolution"
       }
      },
      {"cudnn.SpatialConvolution",
       {cunn = "nn.SpatialConvolutionMM", 
	cudnn = "cudnn.SpatialConvolution"
       }
      },
      {"nn.SpatialFullConvolution",
       {cunn = "nn.SpatialFullConvolution", 
	cudnn = "cudnn.SpatialFullConvolution"
       }
      },
      {"cudnn.SpatialFullConvolution",
       {cunn = "nn.SpatialFullConvolution", 
	cudnn = "cudnn.SpatialFullConvolution"
       }
      }
   }
   if backend:len() == 0 then
      backend = srcnn.backend(old_model)
   end
   local new_model = srcnn.create(model, backend, srcnn.color(old_model))
   for k = 1, #targets do
      local weight_from = old_model:findModules(targets[k][1])
      local weight_to = new_model:findModules(targets[k][2][backend])
      if #weight_from > 0 then
	 if #weight_from ~= #weight_to then
	    error(targets[k][1] .. ": weight_from: " .. #weight_from .. ", weight_to: " .. #weight_to)
	 end
	 for i = 1, #weight_from do
	    local from = weight_from[i]
	    local to = weight_to[i]
	    
	    if to.weight then
	       to.weight:copy(from.weight)
	    end
	    if to.bias then
	       to.bias:copy(from.bias)
	    end
	 end
      end
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
cmd:option("-backend", "", 'Specify the CUDA backend (cunn|cudnn)')
cmd:option("-model", "vgg_7", 'Specify the model architecture (vgg_7|vgg_12|upconv_7|upconv_8_4x|dilated_7)')
cmd:option("-iformat", "ascii", 'Specify the input format (ascii|binary)')
cmd:option("-oformat", "ascii", 'Specify the output format (ascii|binary)')

local opt = cmd:parse(arg)
if not path.isfile(opt.i) then
   cmd:help()
   os.exit(-1)
end
local old_model = torch.load(opt.i, opt.iformat)
local new_model = rebuild(old_model, opt.model, opt.backend)
torch.save(opt.o, new_model, opt.oformat)
