require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "lib", "?.lua;") .. package.path
require 'sys'
require 'w2nn'
local iproc = require 'iproc'
local reconstruct = require 'reconstruct'
local image_loader = require 'image_loader'
local alpha_util = require 'alpha_util'

torch.setdefaulttensortype('torch.FloatTensor')

local function format_output(opt, src, no)
   no = no or 1
   local name = path.basename(src)
   local e = path.extension(name)
   local basename = name:sub(0, name:len() - e:len())
   
   if opt.o == "(auto)" then
      return path.join(path.dirname(src), string.format("%s_%s.png", basename, opt.m))
   else
      local basename_pos = opt.o:find("%%s")
      local no_pos = opt.o:find("%%%d*d")
      if basename_pos ~= nil and no_pos ~= nil then
	 if basename_pos < no_pos then
	    return string.format(opt.o, basename, no)
	 else
	    return string.format(opt.o, no, basename)
	 end
      elseif basename_pos ~= nil then
	 return string.format(opt.o, basename)
      elseif no_pos ~= nil then
	 return string.format(opt.o, no)
      else
	 return opt.o
      end
   end
end

local function convert_image(opt)
   local x, meta = image_loader.load_float(opt.i)
   local alpha = meta.alpha
   local new_x = nil
   local scale_f, image_f

   if opt.tta == 1 then
      scale_f = function(model, scale, x, block_size, batch_size)
	 return reconstruct.scale_tta(model, opt.tta_level,
				      scale, x, block_size, batch_size)
      end
      image_f = function(model, x, block_size, batch_size)
	 return reconstruct.image_tta(model, opt.tta_level,
				      x, block_size, batch_size)
      end
   else
      scale_f = reconstruct.scale
      image_f = reconstruct.image
   end
   opt.o = format_output(opt, opt.i)
   if opt.m == "noise" then
      local model_path = path.join(opt.model_dir, ("noise%d_model.t7"):format(opt.noise_level))
      local model = torch.load(model_path, "ascii")
      if not model then
	 error("Load Error: " .. model_path)
      end
      local t = sys.clock()
      new_x = image_f(model, x, opt.crop_size, opt.batch_size)
      new_x = alpha_util.composite(new_x, alpha)
      print(opt.o .. ": " .. (sys.clock() - t) .. " sec")
   elseif opt.m == "scale" then
      local model_path = path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale))
      local model = torch.load(model_path, "ascii")
      if not model then
	 error("Load Error: " .. model_path)
      end
      local t = sys.clock()
      x = alpha_util.make_border(x, alpha, reconstruct.offset_size(model))
      new_x = scale_f(model, opt.scale, x, opt.crop_size, opt.batch_size, opt.batch_size)
      new_x = alpha_util.composite(new_x, alpha, model)
      print(opt.o .. ": " .. (sys.clock() - t) .. " sec")
   elseif opt.m == "noise_scale" then
      local model_path = path.join(opt.model_dir, ("noise%d_scale%.1fx_model.t7"):format(opt.noise_level, opt.scale))
      if path.exists(model_path) then
	 local scale_model_path = path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale))
	 local t, scale_model = pcall(torch.load, scale_model_path, "ascii")
	 local model = torch.load(model_path, "ascii")
	 if not t then
	    scale_model = model
	 end
	 local t = sys.clock()
	 x = alpha_util.make_border(x, alpha, reconstruct.offset_size(scale_model))
	 new_x = scale_f(model, opt.scale, x, opt.crop_size, opt.batch_size)
	 new_x = alpha_util.composite(new_x, alpha, scale_model)
	 print(opt.o .. ": " .. (sys.clock() - t) .. " sec")
      else
	 local noise_model_path = path.join(opt.model_dir, ("noise%d_model.t7"):format(opt.noise_level))
	 local noise_model = torch.load(noise_model_path, "ascii")
	 local scale_model_path = path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale))
	 local scale_model = torch.load(scale_model_path, "ascii")
	 local t = sys.clock()
	 x = alpha_util.make_border(x, alpha, reconstruct.offset_size(scale_model))
	 x = image_f(noise_model, x, opt.crop_size, opt.batch_size)
	 new_x = scale_f(scale_model, opt.scale, x, opt.crop_size, opt.batch_size)
	 new_x = alpha_util.composite(new_x, alpha, scale_model)
	 print(opt.o .. ": " .. (sys.clock() - t) .. " sec")
      end
   else
      error("undefined method:" .. opt.method)
   end
   image_loader.save_png(opt.o, new_x, tablex.update({depth = opt.depth, inplace = true}, meta))
end
local function convert_frames(opt)
   local model_path, scale_model, t
   local noise_scale_model = {}
   local noise_model = {}
   local scale_f, image_f
   if opt.tta == 1 then
      scale_f = function(model, scale, x, block_size, batch_size)
	 return reconstruct.scale_tta(model, opt.tta_level,
				      scale, x, block_size, batch_size)
      end
      image_f = function(model, x, block_size, batch_size)
	 return reconstruct.image_tta(model, opt.tta_level,
				      x, block_size, batch_size)
      end
   else
      scale_f = reconstruct.scale
      image_f = reconstruct.image
   end
   if opt.m == "scale" then
      model_path = path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale))
      scale_model = torch.load(model_path, "ascii")
   elseif opt.m == "noise" then
      model_path = path.join(opt.model_dir, string.format("noise%d_model.t7", opt.noise_level))
      noise_model[opt.noise_level] = torch.load(model_path, "ascii")
   elseif opt.m == "noise_scale" then
      local model_path = path.join(opt.model_dir, ("noise%d_scale%.1fx_model.t7"):format(opt.noise_level, opt.scale))
      if path.exists(model_path) then
	 noise_scale_model[opt.noise_level] = torch.load(model_path, "ascii")
	 model_path = path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale))
	 t, scale_model = pcall(torch.load, model_path, "ascii")
	 if not t then
	    scale_model = noise_scale_model[opt.noise_level]
	 end
      else
	 model_path = path.join(opt.model_dir, ("scale%.1fx_model.t7"):format(opt.scale))
	 scale_model = torch.load(model_path, "ascii")
	 model_path = path.join(opt.model_dir, string.format("noise%d_model.t7", opt.noise_level))
	 noise_model[opt.noise_level] = torch.load(model_path, "ascii")
      end
   end
   local fp = io.open(opt.l)
   if not fp then
      error("Open Error: " .. opt.l)
   end
   local count = 0
   local lines = {}
   for line in fp:lines() do
      table.insert(lines, line)
   end
   fp:close()
   for i = 1, #lines do
      local output = format_output(opt, lines[i], i)
      if opt.resume == 0 or path.exists(output) == false then
	 local x, meta = image_loader.load_float(lines[i])
	 local alpha = meta.alpha
	 local new_x = nil
	 if opt.m == "noise" then
	    new_x = image_f(noise_model[opt.noise_level], x, opt.crop_size, opt.batch_size)
	    new_x = alpha_util.composite(new_x, alpha)
	 elseif opt.m == "scale" then
	    x = alpha_util.make_border(x, alpha, reconstruct.offset_size(scale_model))
	    new_x = scale_f(scale_model, opt.scale, x, opt.crop_size, opt.batch_size)
	    new_x = alpha_util.composite(new_x, alpha, scale_model)
	 elseif opt.m == "noise_scale" then
	    x = alpha_util.make_border(x, alpha, reconstruct.offset_size(scale_model))
	    if noise_scale_model[opt.noise_level] then
	       new_x = scale_f(noise_scale_model[opt.noise_level], opt.scale, x, opt.crop_size, opt.batch_size)
	    else
	       x = image_f(noise_model[opt.noise_level], x, opt.crop_size, opt.batch_size)
	       new_x = scale_f(scale_model, opt.scale, x, opt.crop_size, opt.batch_size)
	    end
	    new_x = alpha_util.composite(new_x, alpha, scale_model)
	 else
	    error("undefined method:" .. opt.method)
	 end
	 image_loader.save_png(output, new_x, 
			       tablex.update({depth = opt.depth, inplace = true}, meta))
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
   cmd:option("-i", "images/miku_small.png", 'path to input image')
   cmd:option("-l", "", 'path to image-list.txt')
   cmd:option("-scale", 2, 'scale factor')
   cmd:option("-o", "(auto)", 'path to output file')
   cmd:option("-depth", 8, 'bit-depth of the output image (8|16)')
   cmd:option("-model_dir", "./models/upconv_7/art", 'path to model directory')
   cmd:option("-m", "noise_scale", 'method (noise|scale|noise_scale)')
   cmd:option("-noise_level", 1, '(1|2|3)')
   cmd:option("-crop_size", 128, 'patch size per process')
   cmd:option("-batch_size", 1, 'batch_size')
   cmd:option("-resume", 0, "skip existing files (0|1)")
   cmd:option("-thread", -1, "number of CPU threads")
   cmd:option("-tta", 0, '8x slower and slightly high quality (0|1)')
   cmd:option("-tta_level", 8, 'TTA level (2|4|8)')
   
   local opt = cmd:parse(arg)
   if opt.thread > 0 then
      torch.setnumthreads(opt.thread)
   end
   if cudnn then
      cudnn.fastest = true
      cudnn.benchmark = false
   end
   
   if string.len(opt.l) == 0 then
      convert_image(opt)
   else
      convert_frames(opt)
   end
end
waifu2x()
