require 'cudnn'
require 'sys'
require 'pl'
require './lib/LeakyReLU'

local iproc = require './lib/iproc'
local reconstract = require './lib/reconstract'
local image_loader = require './lib/image_loader'

local BLOCK_OFFSET = 7

torch.setdefaulttensortype('torch.FloatTensor')

local function waifu2x()
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text("waifu2x")
   cmd:text("Options:")
   cmd:option("-i", "images/miku_small.png", 'path of input image')
   cmd:option("-o", "(auto)", 'path of output')
   cmd:option("-model_dir", "./models", 'model directory')
   cmd:option("-m", "noise_scale", 'method (noise|scale|noise_scale)')
   cmd:option("-noise_level", 1, '(1|2)')
   cmd:option("-crop_size", 128, 'crop size')
   local opt = cmd:parse(arg)   
   if opt.o == "(auto)" then
      local name = path.basename(opt.i)
      local e = path.extension(name)
      local base = name:sub(0, name:len() - e:len())
      opt.o = path.join(path.dirname(opt.i), string.format("%s(%s).png", base, opt.m))
   end
   
   local x = image_loader.load_float(opt.i)
   local new_x = nil
   local t = sys.clock()
   if opt.m == "noise" then
      local model = torch.load(path.join(opt.model_dir,
					 ("noise%d_model.t7"):format(opt.noise_level)), "ascii")
      model:evaluate()
      new_x = reconstract(model, x, BLOCK_OFFSET)
   elseif opt.m == "scale" then
      local model = torch.load(path.join(opt.model_dir, "scale2.0x_model.t7"), "ascii")
      model:evaluate()
      x = iproc.scale(x, x:size(3) * 2.0, x:size(2) * 2.0)
      new_x = reconstract(model, x, BLOCK_OFFSET)
   elseif opt.m == "noise_scale" then
      local noise_model = torch.load(path.join(opt.model_dir,
					       ("noise%d_model.t7"):format(opt.noise_level)), "ascii")
      local scale_model = torch.load(path.join(opt.model_dir, "scale2.0x_model.t7"), "ascii")

      noise_model:evaluate()
      scale_model:evaluate()
      x = reconstract(noise_model, x, BLOCK_OFFSET)
      x = iproc.scale(x, x:size(3) * 2.0, x:size(2) * 2.0)
      new_x = reconstract(scale_model, x, BLOCK_OFFSET)
   else
      error("undefined method:" .. opt.method)
   end
   image.save(opt.o, new_x)
   print(opt.o .. ": " .. (sys.clock() - t) .. " sec")
end
waifu2x()
