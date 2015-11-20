require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
require 'xlua'
require 'w2nn'
local iproc = require 'iproc'
local reconstruct = require 'reconstruct'
local image_loader = require 'image_loader'
local gm = require 'graphicsmagick'

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x-benchmark")
cmd:text("Options:")

cmd:option("-dir", "./data/test", 'test image directory')
cmd:option("-model1_dir", "./models/anime_style_art_rgb", 'model1 directory')
cmd:option("-model2_dir", "", 'model2 directory (optional)')
cmd:option("-method", "scale", '(scale|noise)')
cmd:option("-filter", "Box", "downscaling filter (Box|Jinc)")
cmd:option("-color", "rgb", '(rgb|y)')
cmd:option("-noise_level", 1, 'model noise level')
cmd:option("-jpeg_quality", 75, 'jpeg quality')
cmd:option("-jpeg_times", 1, 'jpeg compression times')
cmd:option("-jpeg_quality_down", 5, 'value of jpeg quality to decrease each times')

local opt = cmd:parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
if cudnn then
   cudnn.fastest = true
   cudnn.benchmark = false
end

local function MSE(x1, x2)
   return (x1 - x2):pow(2):mean()
end
local function YMSE(x1, x2)
   local x1_2 = image.rgb2y(x1)
   local x2_2 = image.rgb2y(x2)
   return (x1_2 - x2_2):pow(2):mean()
end
local function PSNR(x1, x2)
   local mse = MSE(x1, x2)
   return 10 * math.log10(1.0 / mse)
end
local function YPSNR(x1, x2)
   local mse = YMSE(x1, x2)
   return 10 * math.log10(1.0 / mse)
end

local function transform_jpeg(x, opt)
   for i = 1, opt.jpeg_times do
      jpeg = gm.Image(x, "RGB", "DHW")
      jpeg:format("jpeg")
      jpeg:samplingFactors({1.0, 1.0, 1.0})
      blob, len = jpeg:toBlob(opt.jpeg_quality - (i - 1) * opt.jpeg_quality_down)
      jpeg:fromBlob(blob, len)
      x = jpeg:toTensor("byte", "RGB", "DHW")
   end
   return x
end
local function baseline_scale(x, filter)
   return iproc.scale(x,
		      x:size(3) * 2.0,
		      x:size(2) * 2.0,
		      filter)
end
local function transform_scale(x, opt)
   return iproc.scale(x,
		      x:size(3) * 0.5,
		      x:size(2) * 0.5,
		      opt.filter)
end

local function benchmark(opt, x, input_func, model1, model2)
   local model1_mse = 0
   local model2_mse = 0
   local baseline_mse = 0
   local model1_psnr = 0
   local model2_psnr = 0
   local baseline_psnr = 0
   
   for i = 1, #x do
      local ground_truth = x[i]
      local input, model1_output, model2_output, baseline_output

      input = input_func(ground_truth, opt)
      t = sys.clock()
      if input:size(3) == ground_truth:size(3) then
	 model1_output = reconstruct.image(model1, input)
	 if model2 then
	    model2_output = reconstruct.image(model2, input)
	 end
      else
	 model1_output = reconstruct.scale(model1, 2.0, input)
	 if model2 then
	    model2_output = reconstruct.scale(model2, 2.0, input)
	 end
	 baseline_output = baseline_scale(input, opt.filter)
      end
      if opt.color == "y" then
	 model1_mse = model1_mse + YMSE(ground_truth, model1_output)
	 model1_psnr = model1_psnr + YPSNR(ground_truth, model1_output)
	 if model2 then
	    model2_mse = model2_mse + YMSE(ground_truth, model2_output)
	    model2_psnr = model2_psnr + YPSNR(ground_truth, model2_output)
	 end
	 if baseline_output then
	    baseline_mse = baseline_mse + YMSE(ground_truth, baseline_output)
	    baseline_psnr = baseline_psnr + YPSNR(ground_truth, baseline_output)
	 end
      elseif opt.color == "rgb" then
	 model1_mse = model1_mse + MSE(ground_truth, model1_output)
	 model1_psnr = model1_psnr + PSNR(ground_truth, model1_output)
	 if model2 then
	    model2_mse = model2_mse + MSE(ground_truth, model2_output)
	    model2_psnr = model2_psnr + PSNR(ground_truth, model2_output)
	 end
	 if baseline_output then
	    baseline_mse = baseline_mse + MSE(ground_truth, baseline_output)
	    baseline_psnr = baseline_psnr + PSNR(ground_truth, baseline_output)
	 end
      else
	 error("Unknown color: " .. opt.color)
      end
      if model2 then
	 if baseline_output then
	    io.stdout:write(
	       string.format("%d/%d; baseline_mse=%f, model1_mse=%f, model2_mse=%f, baseline_psnr=%f, model1_psnr=%f, model2_psnr=%f \r",
			     i, #x,
			     baseline_mse / i,
			     model1_mse / i, model2_mse / i,
			     baseline_psnr / i,
			     model1_psnr / i, model2_psnr / i
	    ))
	 else
	    io.stdout:write(
	       string.format("%d/%d; model1_mse=%f, model2_mse=%f, model1_psnr=%f, model2_psnr=%f \r",
			     i, #x,
			     model1_mse / i, model2_mse / i,
			     model1_psnr / i, model2_psnr / i
	    ))
	 end
      else
	 if baseline_output then
	    io.stdout:write(
	       string.format("%d/%d; baseline_mse=%f, model1_mse=%f, baseline_psnr=%f, model1_psnr=%f \r",
			     i, #x,
			     baseline_mse / i, model1_mse / i,
			     baseline_psnr / i, model1_psnr / i
	    ))
	 else
	    io.stdout:write(
	       string.format("%d/%d; model1_mse=%f, model1_psnr=%f \r",
			     i, #x,
			     model1_mse / i, model1_psnr / i
	    ))
	 end
      end
      io.stdout:flush()
   end
   io.stdout:write("\n")
end
local function load_data(test_dir)
   local test_x = {}
   local files = dir.getfiles(test_dir, "*.*")
   for i = 1, #files do
      table.insert(test_x, iproc.crop_mod4(image_loader.load_float(files[i])))
      xlua.progress(i, #files)
   end
   return test_x
end
function load_model(filename)
   return torch.load(filename, "ascii")
end
print(opt)
if opt.method == "scale" then
   local f1 = path.join(opt.model1_dir, "scale2.0x_model.t7")
   local f2 = path.join(opt.model2_dir, "scale2.0x_model.t7")
   local s1, model1 = pcall(load_model, f1)
   local s2, model2 = pcall(load_model, f2)
   if not s1 then
      error("Load error: " .. f1)
   end
   if not s2 then
      model2 = nil
   end
   local test_x = load_data(opt.dir)
   benchmark(opt, test_x, transform_scale, model1, model2)
elseif opt.method == "noise" then
   local f1 = path.join(opt.model1_dir, string.format("noise%d_model.t7", opt.noise_level))
   local f2 = path.join(opt.model2_dir, string.format("noise%d_model.t7", opt.noise_level))
   local s1, model1 = pcall(load_model, f1)
   local s2, model2 = pcall(load_model, f2)
   if not s1 then
      error("Load error: " .. f1)
   end
   if not s2 then
      model2 = nil
   end
   local test_x = load_data(opt.dir)
   benchmark(opt, test_x, transform_jpeg, model1, model2)
end
