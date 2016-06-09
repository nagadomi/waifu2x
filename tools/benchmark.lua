require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
require 'xlua'
require 'w2nn'
local iproc = require 'iproc'
local reconstruct = require 'reconstruct'
local image_loader = require 'image_loader'
local gm = require 'graphicsmagick'
local cjson = require 'cjson'

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x-benchmark")
cmd:text("Options:")

cmd:option("-dir", "./data/test", 'test image directory')
cmd:option("-model1_dir", "./models/anime_style_art_rgb", 'model1 directory')
cmd:option("-model2_dir", "", 'model2 directory (optional)')
cmd:option("-method", "scale", '(scale|noise)')
cmd:option("-filter", "Catrom", "downscaling filter (Box|Lanczos|Catrom(Bicubic))")
cmd:option("-resize_blur", 1.0, 'blur parameter for resize')
cmd:option("-color", "y", '(rgb|y)')
cmd:option("-noise_level", 1, 'model noise level')
cmd:option("-jpeg_quality", 75, 'jpeg quality')
cmd:option("-jpeg_times", 1, 'jpeg compression times')
cmd:option("-jpeg_quality_down", 5, 'value of jpeg quality to decrease each times')
cmd:option("-range_bug", 0, 'Reproducing the dynamic range bug that is caused by MATLAB\'s rgb2ycbcr(1|0)')
cmd:option("-save_image", 0, 'save converted images')
cmd:option("-save_baseline_image", 0, 'save baseline images')
cmd:option("-output_dir", "./", 'output directroy')
cmd:option("-show_progress", 1, 'show progressbar')
cmd:option("-baseline_filter", "Catrom", 'baseline interpolation (Box|Lanczos|Catrom(Bicubic))')
cmd:option("-save_info", 0, 'save score and parameters to benchmark.txt')
cmd:option("-save_all", 0, 'group -save_info, -save_image and -save_baseline_image option')
cmd:option("-thread", -1, 'number of CPU threads')
cmd:option("-tta", 0, 'tta')

local function to_bool(settings, name)
   if settings[name] == 1 then
      settings[name] = true
   else
      settings[name] = false
   end
end
local opt = cmd:parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
if cudnn then
   cudnn.fastest = true
   cudnn.benchmark = false
end
to_bool(opt, "save_all")
to_bool(opt, "tta")
if opt.save_all then
   opt.save_image = true
   opt.save_info = true
   opt.save_baseline_image = true
else
   to_bool(opt, "save_image")
   to_bool(opt, "save_info")
   to_bool(opt, "save_baseline_image")
end
to_bool(opt, "show_progress")
if opt.thread > 0 then
   torch.setnumthreads(tonumber(opt.thread))
end

local function rgb2y_matlab(x)
   local y = torch.Tensor(1, x:size(2), x:size(3)):zero()
   x = iproc.byte2float(x)
   y:add(x[1] * 65.481)
   y:add(x[2] * 128.553)
   y:add(x[3] * 24.966)
   y:add(16.0)
   return y:byte():float()
end

local function RGBMSE(x1, x2)
   x1 = iproc.float2byte(x1):float()
   x2 = iproc.float2byte(x2):float()
   return (x1 - x2):pow(2):mean()
end
local function YMSE(x1, x2)
   if opt.range_bug == 1 then
      local x1_2 = rgb2y_matlab(x1)
      local x2_2 = rgb2y_matlab(x2)
      return (x1_2 - x2_2):pow(2):mean()
   else
      local x1_2 = image.rgb2y(x1):mul(255.0)
      local x2_2 = image.rgb2y(x2):mul(255.0)
      return (x1_2 - x2_2):pow(2):mean()
   end
end
local function MSE(x1, x2, color)
   if color == "y" then
      return YMSE(x1, x2)
   else
      return RGBMSE(x1, x2)
   end
end
local function PSNR(x1, x2, color)
   local mse = math.max(MSE(x1, x2, color), 1)
   return 10 * math.log10((255.0 * 255.0) / mse)
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
   return iproc.byte2float(x)
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
		      opt.filter, opt.resize_blur)
end

local function benchmark(opt, x, input_func, model1, model2)
   local model1_mse = 0
   local model2_mse = 0
   local baseline_mse = 0
   local model1_psnr = 0
   local model2_psnr = 0
   local baseline_psnr = 0
   local scale_f = reconstruct.scale
   local image_f = reconstruct.image
   if opt.tta then
      scale_f = reconstruct.scale_tta
      image_f = reconstruct.image_tta
   end
   
   for i = 1, #x do
      local ground_truth = x[i].image
      local basename = x[i].basename
      
      local input, model1_output, model2_output, baseline_output

      input = input_func(ground_truth, opt)
      t = sys.clock()
      if input:size(3) == ground_truth:size(3) then
	 model1_output = image_f(model1, input)
	 if model2 then
	    model2_output = image_f(model2, input)
	 end
      else
	 model1_output = scale_f(model1, 2.0, input)
	 if model2 then
	    model2_output = scale_f(model2, 2.0, input)
	 end
	 baseline_output = baseline_scale(input, opt.baseline_filter)
      end
      model1_mse = model1_mse + MSE(ground_truth, model1_output, opt.color)
      model1_psnr = model1_psnr + PSNR(ground_truth, model1_output, opt.color)
      if model2 then
	 model2_mse = model2_mse + MSE(ground_truth, model2_output, opt.color)
	 model2_psnr = model2_psnr + PSNR(ground_truth, model2_output, opt.color)
      end
      if baseline_output then
	 baseline_mse = baseline_mse + MSE(ground_truth, baseline_output, opt.color)
	 baseline_psnr = baseline_psnr + PSNR(ground_truth, baseline_output, opt.color)
      end
      if opt.save_image then
	 if opt.save_baseline_image and baseline_output then
	    image.save(path.join(opt.output_dir, string.format("%s_baseline.png", basename)),
		       baseline_output)
	 end
	 if model1_output then
	    image.save(path.join(opt.output_dir, string.format("%s_model1.png", basename)),
		       model1_output)
	 end
	 if model2_output then
	    image.save(path.join(opt.output_dir, string.format("%s_model2.png", basename)),
		       model2_output)
	 end
      end
      if opt.show_progress or i == #x then
	 if model2 then
	    if baseline_output then
	       io.stdout:write(
		  string.format("%d/%d; baseline_rmse=%f, model1_rmse=%f, model2_rmse=%f, baseline_psnr=%f, model1_psnr=%f, model2_psnr=%f \r",
				i, #x,
				math.sqrt(baseline_mse / i),
				math.sqrt(model1_mse / i), math.sqrt(model2_mse / i),
				baseline_psnr / i,
				model1_psnr / i, model2_psnr / i
		  ))
	    else
	       io.stdout:write(
		  string.format("%d/%d; model1_rmse=%f, model2_rmse=%f, model1_psnr=%f, model2_psnr=%f \r",
				i, #x,
				math.sqrt(model1_mse / i), math.sqrt(model2_mse / i),
				model1_psnr / i, model2_psnr / i
		  ))
	    end
	 else
	    if baseline_output then
	       io.stdout:write(
		  string.format("%d/%d; baseline_rmse=%f, model1_rmse=%f, baseline_psnr=%f, model1_psnr=%f \r",
				i, #x,
				math.sqrt(baseline_mse / i), math.sqrt(model1_mse / i),
				baseline_psnr / i, model1_psnr / i
		  ))
	    else
	       io.stdout:write(
		  string.format("%d/%d; model1_rmse=%f, model1_psnr=%f \r",
				i, #x,
				math.sqrt(model1_mse / i), model1_psnr / i
		  ))
	    end
	 end
	 io.stdout:flush()
      end
   end
   if opt.save_info then
      local fp = io.open(path.join(opt.output_dir, "benchmark.txt"), "w")
      fp:write("options : " .. cjson.encode(opt) .. "\n")
      if baseline_psnr > 0 then
	 fp:write(string.format("baseline: RMSE = %.3f, PSNR = %.3f\n",
				math.sqrt(baseline_mse / #x), baseline_psnr / #x))
      end
      if model1_psnr > 0 then
	 fp:write(string.format("model1  : RMSE = %.3f, PSNR = %.3f\n",
				math.sqrt(model1_mse / #x), model1_psnr / #x))
      end
      if model2_psnr > 0 then
	 fp:write(string.format("model2  : RMSE = %.3f, PSNR = %.3f\n",
				math.sqrt(model2_mse / #x), model2_psnr / #x))
      end
      fp:close()
   end
   io.stdout:write("\n")
end
local function load_data(test_dir)
   local test_x = {}
   local files = dir.getfiles(test_dir, "*.*")
   for i = 1, #files do
      local name = path.basename(files[i])
      local e = path.extension(name)
      local base = name:sub(0, name:len() - e:len())
      local img = image_loader.load_float(files[i])
      if img then
	 table.insert(test_x, {image = iproc.crop_mod4(img),
			       basename = base})
      end
      if opt.show_progress then
	 xlua.progress(i, #files)
      end
   end
   return test_x
end
function load_model(filename)
   return torch.load(filename, "ascii")
end
if opt.show_progress then
   print(opt)
end
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
