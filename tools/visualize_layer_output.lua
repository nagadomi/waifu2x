require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
require 'sys'
require 'w2nn'
local iproc = require 'iproc'
local reconstruct = require 'reconstruct'
local image_loader = require 'image_loader'

local CONV_LAYERS = {"nn.SpatialConvolutionMM",
		     "cudnn.SpatialConvolution",
		     "nn.SpatialFullConvolution",
		     "cudnn.SpatialFullConvolution"
}
local ACTIVATION_LAYERS = {"nn.ReLU",
			   "nn.LeakyReLU",
			   "w2nn.LeakyReLU",
			   "cudnn.ReLU",
			   "nn.SoftMax",
			   "cudnn.SoftMax"
}

local function includes(s, a)
   for i = 1, #a do
      if s == a[i] then
	 return true
      end
   end
   return false
end
local function count_conv_layers(seq)
   local count = 0
   for k = 1, #seq.modules do
      local mod = seq.modules[k]
      local name = torch.typename(mod)
      if name == "nn.ConcatTable" or includes(name, CONV_LAYERS) then
	 count = count + 1
      end
   end
   return count
end
local function strip_conv_layers(seq, limit)
   local new_seq = nn.Sequential()
   local count = 0
   for k = 1, #seq.modules do
      local mod = seq.modules[k]
      local name = torch.typename(mod)
      if name == "nn.ConcatTable" or includes(name, CONV_LAYERS) then
	 new_seq:add(mod)
	 count = count + 1
	 if count == limit then
	    if seq.modules[k+1] ~= nil and 
	    includes(torch.typename(seq.modules[k+1]), ACTIVATION_LAYERS) then
	       new_seq:add(seq.modules[k+1])
	    end
	    return new_seq
	 end
      else
	 new_seq:add(mod)
      end
   end
   return new_seq
end
local function save_layer_outputs(x, model, out)
   local count = count_conv_layers(model)
   print("conv layer count", count)
   local output_file = path.join(out, string.format("layer-%d.png", 0))
   image.save(output_file, x)
   print("* save layer output " .. 0 .. ": " .. output_file)
   for i = 1, count do
      output_file = path.join(out, string.format("layer-%d.png", i))
      print("* save layer output " .. i .. ": " .. output_file)
      local test_model = strip_conv_layers(model, i)
      test_model:cuda()
      test_model:evaluate()
      local z = test_model:forward(x:reshape(1, x:size(1), x:size(2), x:size(3)):cuda()):float()
      z = z:reshape(z:size(2), z:size(3), z:size(4)) -- drop batch dim
      z = image.toDisplayTensor({input=z, padding=2})
      image.save(output_file, z)
      collectgarbage()
   end
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x - visualize layer output")
cmd:text("Options:")
cmd:option("-i", "images/miku_small.png", 'path to input image')
cmd:option("-scale", 2, 'scale factor')
cmd:option("-o", "./layer_outputs", 'path to output dir')
cmd:option("-model_dir", "./models/upconv_7/art", 'path to model directory')
cmd:option("-name", "user", 'model name for user method')
cmd:option("-m", "noise_scale", 'method (noise|scale|noise_scale|user)')
cmd:option("-noise_level", 1, '(1|2|3)')
cmd:option("-force_cudnn", 0, 'use cuDNN backend (0|1)')
cmd:option("-gpu", 1, 'Device ID')

local opt = cmd:parse(arg)
cutorch.setDevice(opt.gpu)
opt.force_cudnn = opt.force_cudnn == 1
opt.model_path = path.join(opt.model_dir, string.format("%s_model.t7", opt.name))

local x, meta = image_loader.load_float(opt.i)
if x:size(2) > 256 or x:size(3) > 256 then
   error(string.format("input image is too large: %dx%d", x:size(3), x:size(2)))
end
local model = nil
local new_x = nil

if opt.m == "noise" then
   local model_path = path.join(opt.model_dir, ("noise%d_model.t7"):format(opt.noise_level))
   model = w2nn.load_model(model_path, opt.force_cudnn)
   if not model then
      error("Load Error: " .. model_path)
   end
elseif opt.m == "scale" then
   local model_path = path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale))
   model = w2nn.load_model(model_path, opt.force_cudnn)
   if not model then
      error("Load Error: " .. model_path)
   end
elseif opt.m == "noise_scale" then
   local model_path = path.join(opt.model_dir, ("noise%d_scale%.1fx_model.t7"):format(opt.noise_level, opt.scale))
   model = w2nn.load_model(model_path, opt.force_cudnn)
elseif opt.m == "user" then
   local model_path = opt.model_path
   model = w2nn.load_model(model_path, opt.force_cudnn)
   if not model then
      error("Load Error: " .. model_path)
   end
else
   error("undefined method:" .. opt.method)
end
assert(model ~= nil)
assert(x ~= nil)

dir.makepath(opt.o)
save_layer_outputs(x, model, opt.o)
