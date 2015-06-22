require './lib/portable'
require 'sys'
require 'pl'
require './lib/LeakyReLU'

local iproc = require './lib/iproc'
local reconstruct = require './lib/reconstruct'
local image_loader = require './lib/image_loader'
local BLOCK_OFFSET = 7

torch.setdefaulttensortype('torch.FloatTensor')

local function convert_image(opt)
   local x, alpha = image_loader.load_float(opt.i)
   local new_x = nil
   local t = sys.clock()
   if opt.o == "(auto)" then
      local name = path.basename(opt.i)
      local e = path.extension(name)
      local base = name:sub(0, name:len() - e:len())
      opt.o = path.join(path.dirname(opt.i), string.format("%s(%s).png", base, opt.m))
   end
   if opt.m == "noise" then
      local model = torch.load(path.join(opt.model_dir, ("noise%d_model.t7"):format(opt.noise_level)), "ascii")
      model:evaluate()
      new_x = reconstruct.image(model, x, BLOCK_OFFSET, opt.crop_size)
   elseif opt.m == "scale" then
      local model = torch.load(path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale)), "ascii")
      model:evaluate()
      new_x = reconstruct.scale(model, opt.scale, x, BLOCK_OFFSET, opt.crop_size)
   elseif opt.m == "noise_scale" then
      local noise_model = torch.load(path.join(opt.model_dir, ("noise%d_model.t7"):format(opt.noise_level)), "ascii")
      local scale_model = torch.load(path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale)), "ascii")
      noise_model:evaluate()
      scale_model:evaluate()
      x = reconstruct.image(noise_model, x, BLOCK_OFFSET)
      new_x = reconstruct.scale(scale_model, opt.scale, x, BLOCK_OFFSET, opt.crop_size)
   else
      error("undefined method:" .. opt.method)
   end
   image_loader.save_png(opt.o, new_x, alpha)
   print(opt.o .. ": " .. (sys.clock() - t) .. " sec")
end
local function convert_frames(opt)
   local noise1_model = torch.load(path.join(opt.model_dir, "noise1_model.t7"), "ascii")
   local noise2_model = torch.load(path.join(opt.model_dir, "noise2_model.t7"), "ascii")
   local scale_model = torch.load(path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale)), "ascii")

   noise1_model:evaluate()
   noise2_model:evaluate()
   scale_model:evaluate()
   
   local fp = io.open(opt.l)
   local count = 0
   local lines = {}
   for line in fp:lines() do
      table.insert(lines, line)
   end
   fp:close()
   for i = 1, #lines do
      if opt.resume == 0 or path.exists(string.format(opt.o, i)) == false then
	 local x, alpha = image_loader.load_float(lines[i])
	 local new_x = nil
	 if opt.m == "noise" and opt.noise_level == 1 then
	    new_x = reconstruct.image(noise1_model, x, BLOCK_OFFSET, opt.crop_size)
	 elseif opt.m == "noise" and opt.noise_level == 2 then
	    new_x = reconstruct.image(noise2_model, x, BLOCK_OFFSET)
	 elseif opt.m == "scale" then
	    new_x = reconstruct.scale(scale_model, opt.scale, x, BLOCK_OFFSET, opt.crop_size)
	 elseif opt.m == "noise_scale" and opt.noise_level == 1 then
	    x = reconstruct.image(noise1_model, x, BLOCK_OFFSET)
	    new_x = reconstruct.scale(scale_model, opt.scale, x, BLOCK_OFFSET, opt.crop_size)
	 elseif opt.m == "noise_scale" and opt.noise_level == 2 then
	    x = reconstruct.image(noise2_model, x, BLOCK_OFFSET)
	    new_x = reconstruct.scale(scale_model, opt.scale, x, BLOCK_OFFSET, opt.crop_size)
	 else
	    error("undefined method:" .. opt.method)
	 end
	 local output = nil
	 if opt.o == "(auto)" then
	    local name = path.basename(lines[i])
	    local e = path.extension(name)
	    local base = name:sub(0, name:len() - e:len())
	    output = path.join(path.dirname(opt.i), string.format("%s(%s).png", base, opt.m))
	 else
	    output = string.format(opt.o, i)
	 end
	 image_loader.save_png(output, new_x, alpha)
	 xlua.progress(i, #lines)
	 if i % 10 == 0 then
	    collectgarbage()
	 end
      else
	 xlua.progress(i, #lines)
      end
   end
end

local function waifu2x()
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text("waifu2x")
   cmd:text("Options:")
   cmd:option("-i", "images/miku_small.png", 'path of the input image')
   cmd:option("-l", "", 'path of the image-list')
   cmd:option("-scale", 2, 'scale factor')
   cmd:option("-o", "(auto)", 'path of the output file')
   cmd:option("-model_dir", "./models/anime_style_art_rgb", 'model directory')
   cmd:option("-m", "noise_scale", 'method (noise|scale|noise_scale)')
   cmd:option("-noise_level", 1, '(1|2)')
   cmd:option("-crop_size", 128, 'patch size per process')
   cmd:option("-resume", 0, "skip existing files (0|1)")
   
   local opt = cmd:parse(arg)
   if string.len(opt.l) == 0 then
      convert_image(opt)
   else
      convert_frames(opt)
   end
end
waifu2x()
