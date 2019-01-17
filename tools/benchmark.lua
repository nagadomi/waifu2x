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
cmd:option("-file", "", 'test image file list')
cmd:option("-model1_dir", "./models/anime_style_art_rgb", 'model1 directory')
cmd:option("-model2_dir", "", 'model2 directory (optional)')
cmd:option("-method", "scale", '(scale|noise|noise_scale|user|diff|scale4)')
cmd:option("-filter", "Catrom", "downscaling filter (Box|Lanczos|Catrom(Bicubic))")
cmd:option("-resize_blur", 1.0, 'blur parameter for resize')
cmd:option("-color", "y", '(rgb|y|r|g|b)')
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
cmd:option("-tta", 0, 'use tta')
cmd:option("-tta_level", 8, 'tta level')
cmd:option("-crop_size", 256, 'patch size per process')
cmd:option("-batch_size", 1, 'batch_size')
cmd:option("-force_cudnn", 0, 'use cuDNN backend')
cmd:option("-yuv420", 0, 'use yuv420 jpeg')
cmd:option("-name", "", 'model name for user method')
cmd:option("-x_dir", "", 'input image for user method')
cmd:option("-y_dir", "", 'groundtruth image for user method. filename must be the same as x_dir')
cmd:option("-x_file", "", 'input image for user method')
cmd:option("-y_file", "", 'groundtruth image for user method. filename must be the same as x_file')
cmd:option("-border", 0, 'border px that will removed')
cmd:option("-metric", "", '(jaccard)')

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
   cudnn.benchmark = true
end
to_bool(opt, "force_cudnn")
to_bool(opt, "yuv420")
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
if opt.output_dir:len() > 0 then
   dir.makepath(opt.output_dir)
end

-- patch for lua52
if not math.log10 then
   math.log10 = function(x) return math.log(x, 10) end
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
local function CHMSE(x1, x2, ch)
   x1 = iproc.float2byte(x1):float()
   x2 = iproc.float2byte(x2):float()
   return (x1[ch] - x2[ch]):pow(2):mean()
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
   elseif color == "r" then
      return CHMSE(x1, x2, 1)
   elseif color == "g" then
      return CHMSE(x1, x2, 2)
   elseif color == "b" then
      return CHMSE(x1, x2, 3)
   else
      return RGBMSE(x1, x2)
   end
end
local function PSNR(x1, x2, color)
   local mse = math.max(MSE(x1, x2, color), 1)
   return 10 * math.log10((255.0 * 255.0) / mse)
end
local function MSE2PSNR(mse)
   return 10 * math.log10((255.0 * 255.0) / math.max(mse, 1))
end
local function transform_jpeg(x, opt)
   for i = 1, opt.jpeg_times do
      jpeg = gm.Image(x, "RGB", "DHW")
      jpeg:format("jpeg")
      if opt.yuv420 then
	 jpeg:samplingFactors({2.0, 1.0, 1.0})
      else
	 jpeg:samplingFactors({1.0, 1.0, 1.0})
      end
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
local function baseline_scale4(x, filter)
   return iproc.scale(x,
		      x:size(3) * 4.0,
		      x:size(2) * 4.0,
		      filter)
end
local function transform_scale(x, opt)
   return iproc.scale(x,
		      x:size(3) * 0.5,
		      x:size(2) * 0.5,
		      opt.filter, opt.resize_blur)
end
local function transform_scale4(x, opt)
   return iproc.scale(x,
		      x:size(3) * 0.25,
		      x:size(2) * 0.25,
		      opt.filter, opt.resize_blur)
end

local function transform_scale_jpeg(x, opt)
   x = iproc.scale(x,
		   x:size(3) * 0.5,
		   x:size(2) * 0.5,
		   opt.filter, opt.resize_blur)
   for i = 1, opt.jpeg_times do
      jpeg = gm.Image(x, "RGB", "DHW")
      jpeg:format("jpeg")
      if opt.yuv420 then
	 jpeg:samplingFactors({2.0, 1.0, 1.0})
      else
	 jpeg:samplingFactors({1.0, 1.0, 1.0})
      end
      blob, len = jpeg:toBlob(opt.jpeg_quality - (i - 1) * opt.jpeg_quality_down)
      jpeg:fromBlob(blob, len)
      x = jpeg:toTensor("byte", "RGB", "DHW")
   end
   return iproc.byte2float(x)
end
local function remove_border(x, border)
   return iproc.crop(x,
		     border, border,
		     x:size(3) - border,
		     x:size(2) - border)
end
local function create_metric(metric)
   if metric and metric:len() > 0 then
      if metric == "jaccard" then
	 return {
	    name = "jaccard", 
	    func = function (a, b) 
	       local ga = iproc.rgb2y(a)
	       local gb = iproc.rgb2y(b)
	       local ba = torch.Tensor():resizeAs(ga)
	       local bb = torch.Tensor():resizeAs(gb)
	       ba:zero()
	       bb:zero()
	       ba[torch.gt(ga, 0.5)] = 1.0
	       bb[torch.gt(gb, 0.5)] = 1.0
	       local num_a = ba:sum()
	       local num_b = bb:sum()
	       local a_and_b  = ba:cmul(bb):sum()
	       local abab = (num_a + num_b - a_and_b)
	       if abab > 0 then
		  return (a_and_b / abab)
	       else
		  return 1
	       end
	 end}
      else
	 error("unknown metric: " .. metric)
      end
   else
      return nil
   end
end
local function benchmark(opt, x, model1, model2)
   local mse1, mse2, am1, am2
   local won = {0, 0}
   local model1_mse = 0
   local model2_mse = 0
   local baseline_mse = 0
   local model1_psnr = 0
   local model2_psnr = 0
   local baseline_psnr = 0
   local model1_time = 0
   local model2_time = 0
   local scale_f = reconstruct.scale
   local image_f = reconstruct.image
   local detail_fp = nil
   local am = nil
   local model1_am = 0
   local model2_am = 0

   if opt.method == "user" or opt.method == "diff" then
      am = create_metric(opt.metric)
   end
   if opt.save_info then
      detail_fp = io.open(path.join(opt.output_dir, "benchmark_details.txt"), "w")
   end
   if opt.tta then
      scale_f = function(model, scale, x, block_size, batch_size)
	 return reconstruct.scale_tta(model, opt.tta_level,
				      scale, x, block_size, batch_size)
      end
      image_f = function(model, x, block_size, batch_size)
	 return reconstruct.image_tta(model, opt.tta_level,
				      x, block_size, batch_size)
      end
   end

   for i = 1, #x do
      if i % 10 == 0 then
	 collectgarbage()
      end
      local basename = x[i].basename
      local input, model1_output, model2_output, baseline_output, ground_truth

      if opt.method == "scale" then
	 input = transform_scale(iproc.byte2float(x[i].y), opt)
	 ground_truth = iproc.byte2float(x[i].y)

	 if opt.force_cudnn and i == 1 then -- run cuDNN benchmark first
	    model1_output = scale_f(model1, 2.0, input, opt.crop_size, opt.batch_size)
	    if model2 then
	       model2_output = scale_f(model2, 2.0, input, opt.crop_size, opt.batch_size)
	    end
	 end
	 t = sys.clock()
	 model1_output = scale_f(model1, 2.0, input, opt.crop_size, opt.batch_size)
	 model1_time = model1_time + (sys.clock() - t)
	 if model2 then
	    t = sys.clock()
	    model2_output = scale_f(model2, 2.0, input, opt.crop_size, opt.batch_size)
	    model2_time = model2_time + (sys.clock() - t)
	 end
	 baseline_output = baseline_scale(input, opt.baseline_filter)
      elseif opt.method == "scale4" then
	 input = transform_scale4(iproc.byte2float(x[i].y), opt)
	 ground_truth = iproc.byte2float(x[i].y)
	 if opt.force_cudnn and i == 1 then -- run cuDNN benchmark first
	    model1_output = scale_f(model1, 2.0, input, opt.crop_size, opt.batch_size)
	    if model2 then
	       model2_output = scale_f(model2, 2.0, input, opt.crop_size, opt.batch_size)
	    end
	 end
	 t = sys.clock()
	 model1_output = scale_f(model1, 2.0, input, opt.crop_size, opt.batch_size)
	 model1_output = scale_f(model1, 2.0, model1_output, opt.crop_size, opt.batch_size)
	 model1_time = model1_time + (sys.clock() - t)
	 if model2 then
	    t = sys.clock()
	    model2_output = scale_f(model2, 2.0, input, opt.crop_size, opt.batch_size)
	    model2_output = scale_f(model2, 2.0, model2_output, opt.crop_size, opt.batch_size)
	    model2_time = model2_time + (sys.clock() - t)
	 end
	 baseline_output = baseline_scale4(input, opt.baseline_filter)
      elseif opt.method == "noise" then
	 input = transform_jpeg(iproc.byte2float(x[i].y), opt)
	 ground_truth = iproc.byte2float(x[i].y)

	 if opt.force_cudnn and i == 1 then
	    model1_output = image_f(model1, input, opt.crop_size, opt.batch_size)
	    if model2 then
	       model2_output = image_f(model2, input, opt.crop_size, opt.batch_size)
	    end
	 end
	 t = sys.clock()
	 model1_output = image_f(model1, input, opt.crop_size, opt.batch_size)
	 model1_time = model1_time + (sys.clock() - t)
	 if model2 then
	    t = sys.clock()
	    model2_output = image_f(model2, input, opt.crop_size, opt.batch_size)
	    model2_time = model2_time + (sys.clock() - t)
	 end
	 baseline_output = input
      elseif opt.method == "noise_scale" then
	 input = transform_scale_jpeg(iproc.byte2float(x[i].y), opt)
	 ground_truth = iproc.byte2float(x[i].y)

	 if opt.force_cudnn and i == 1 then
	    if model1.noise_scale_model then
	       model1_output = scale_f(model1.noise_scale_model, 2.0,
				       input, opt.crop_size, opt.batch_size)
	    else
	       if model1.noise_model then
	       model1_output = image_f(model1.noise_model, input, opt.crop_size, opt.batch_size)
	       else
		  model1_output = input
	       end
	       model1_output = scale_f(model1.scale_model, 2.0, model1_output,
				       opt.crop_size, opt.batch_size)
	    end
	    if model2 then
	       if model2.noise_scale_model then
		  model2_output = scale_f(model2.noise_scale_model, 2.0,
					  input, opt.crop_size, opt.batch_size)
	       else
		  if model2.noise_model then
		     model2_output = image_f(model2.noise_model, input,
					     opt.crop_size, opt.batch_size)
		  else
		     model2_output = input
		  end
		  model2_output = scale_f(model2.scale_model, 2.0, model2_output,
				       opt.crop_size, opt.batch_size)
	       end
	    end
	 end
	 t = sys.clock()
	 if model1.noise_scale_model then
	    model1_output = scale_f(model1.noise_scale_model, 2.0,
				    input, opt.crop_size, opt.batch_size)
	 else
	    if model1.noise_model then
	       model1_output = image_f(model1.noise_model, input, opt.crop_size, opt.batch_size)
	    else
	       model1_output = input
	    end
	    model1_output = scale_f(model1.scale_model, 2.0, model1_output,
				    opt.crop_size, opt.batch_size)
	 end
	 model1_time = model1_time + (sys.clock() - t)

	 if model2 then
	    t = sys.clock()
	    if model2.noise_scale_model then
	       model2_output = scale_f(model2.noise_scale_model, 2.0,
				       input, opt.crop_size, opt.batch_size)
	    else
	       if model2.noise_model then
		  model2_output = image_f(model2.noise_model, input,
					  opt.crop_size, opt.batch_size)
	       else
		  model2_output = input
	       end
	       model2_output = scale_f(model2.scale_model, 2.0, model2_output,
				       opt.crop_size, opt.batch_size)
	    end
	    model2_time = model2_time + (sys.clock() - t)
	 end
	 baseline_output = baseline_scale(input, opt.baseline_filter)
      elseif opt.method == "user" then
	 input = iproc.byte2float(x[i].x)
	 ground_truth = iproc.byte2float(x[i].y)
	 local y_scale = ground_truth:size(2) / input:size(2)
	 if y_scale > 1 then
	    if opt.force_cudnn and i == 1 then
	       model1_output = scale_f(model1, y_scale, input, opt.crop_size, opt.batch_size)
	       if model2 then
		  model2_output = scale_f(model2, y_scale, input, opt.crop_size, opt.batch_size)
	       end
	    end
	    t = sys.clock()
	    model1_output = scale_f(model1, y_scale, input, opt.crop_size, opt.batch_size)
	    model1_time = model1_time + (sys.clock() - t)
	    if model2 then
	       t = sys.clock()
	       model2_output = scale_f(model2, y_scale, input, opt.crop_size, opt.batch_size)
	       model2_time = model2_time + (sys.clock() - t)
	    end
	 else
	    if opt.force_cudnn and i == 1 then
	       model1_output = image_f(model1, input, opt.crop_size, opt.batch_size)
	       if model2 then
		  model2_output = image_f(model2, input, opt.crop_size, opt.batch_size)
	       end
	    end
	    t = sys.clock()
	    model1_output = image_f(model1, input, opt.crop_size, opt.batch_size)
	    model1_time = model1_time + (sys.clock() - t)
	    if model2 then
	       t = sys.clock()
	       model2_output = image_f(model2, input, opt.crop_size, opt.batch_size)
	       model2_time = model2_time + (sys.clock() - t)
	    end
	 end
      elseif opt.method == "diff" then
	 input = iproc.byte2float(x[i].x)
	 ground_truth = iproc.byte2float(x[i].y)
	 model1_output = input
      end
      if opt.border > 0 then
	 ground_truth = remove_border(ground_truth, opt.border)
	 model1_output = remove_border(model1_output, opt.border)
      end
      if am then
	 am1 = am.func(ground_truth, model1_output)
	 model1_am = model1_am + am1
      else
	 mse1 = MSE(ground_truth, model1_output, opt.color)
	 model1_mse = model1_mse + mse1
	 model1_psnr = model1_psnr + MSE2PSNR(mse1)
      end
      local won_model = 1
      if model2 then
	 if opt.border > 0 then
	    model2_output = remove_border(model2_output, opt.border)
	 end
	 if am then
	    am2 = am.func(ground_truth, model2_output)
	    model2_am = model2_am + am2
	 else
	    mse2 = MSE(ground_truth, model2_output, opt.color)
	    model2_mse = model2_mse + mse2
	    model2_psnr = model2_psnr + MSE2PSNR(mse2)
	 end
	 if am then
	    if am1 < am2 then
	       won[1] = won[1] + 1
	    elseif am1 > am2 then
	       won[2] = won[2] + 1
	       won_model = 2
	    end
	 else
	    if mse1 < mse2 then
	       won[1] = won[1] + 1
	    elseif mse1 > mse2 then
	       won[2] = won[2] + 1
	       won_model = 2
	    end
	 end
	 if detail_fp then
	    if am then
	       detail_fp:write(string.format("%s,%f,%d\n", x[i].basename, am1, am2, won_model))
	    else
	       detail_fp:write(string.format("%s,%f,%f,%d\n", x[i].basename,
					     MSE2PSNR(mse1), MSE2PSNR(mse2), won_model))
	    end
	 end
      else
	 if detail_fp then
	    if am then
	       detail_fp:write(string.format("%s,%f\n", x[i].basename, am1))
	    else
	       detail_fp:write(string.format("%s,%f\n", x[i].basename, MSE2PSNR(mse1)))
	    end
	 end
      end
      if baseline_output then
	 baseline_output = remove_border(baseline_output, opt.border)
	 mse = MSE(ground_truth, baseline_output, opt.color)
	 baseline_mse = baseline_mse + mse
	 baseline_psnr = baseline_psnr + MSE2PSNR(mse)
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
	 if am then
	    if model2 then
	       io.stdout:write(
		  string.format("%d/%d; model1_time=%.2f, model2_time=%.2f, model1_%s=%.3f, model2_%s=%.3f \r",
				i, #x,
				model1_time,
				model2_time,
				am.name, model1_am / i, am.name, model2_am / i
	       ))
	    else
	       io.stdout:write(
		  string.format("%d/%d; model1_time=%.2f, model1_%s=%.3f \r",
				i, #x,
				model1_time,
				am.name, model1_am / i
	       ))
	    end
	 else
	    if model2 then
	       if baseline_output then
		  io.stdout:write(
		     string.format("%d/%d; model1_time=%.2f, model2_time=%.2f, baseline_rmse=%.3f, model1_rmse=%.3f, model2_rmse=%.3f, baseline_psnr=%.3f, model1_psnr=%.3f, model2_psnr=%.3f, model1_won=%d, model2_won=%d \r",
				   i, #x,
				   model1_time,
				   model2_time,
				   math.sqrt(baseline_mse / i),
				   math.sqrt(model1_mse / i), math.sqrt(model2_mse / i),
				   baseline_psnr / i,
				   model1_psnr / i, model2_psnr / i,
				   won[1], won[2]
		  ))
	       else
		  io.stdout:write(
		     string.format("%d/%d; model1_time=%.2f, model2_time=%.2f, model1_rmse=%.3f, model2_rmse=%.3f, model1_psnr=%.3f, model2_psnr=%.3f, model1_own=%d, model2_won=%d \r",
				   i, #x,
				   model1_time,
				   model2_time,
				   math.sqrt(model1_mse / i), math.sqrt(model2_mse / i),
				   model1_psnr / i, model2_psnr / i,
				   won[1], won[2]
		  ))
	       end
	    else
	       if baseline_output then
		  io.stdout:write(
		     string.format("%d/%d; model1_time=%.2f, baseline_rmse=%.3f, model1_rmse=%.3f, baseline_psnr=%.3f, model1_psnr=%.3f \r",
				   i, #x,
				   model1_time,
				   math.sqrt(baseline_mse / i), math.sqrt(model1_mse / i),
				   baseline_psnr / i, model1_psnr / i
		  ))
	       else
		  io.stdout:write(
		     string.format("%d/%d; model1_time=%.2f, model1_rmse=%.3f, model1_psnr=%.3f \r",
				   i, #x,
				   model1_time,
				   math.sqrt(model1_mse / i), model1_psnr / i
		  ))
	       end
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
	 fp:write(string.format("model1  : RMSE = %.3f, PSNR = %.3f, evaluation time = %.3f\n",
				math.sqrt(model1_mse / #x), model1_psnr / #x, model1_time))
      end
      if model2_psnr > 0 then
	 fp:write(string.format("model2  : RMSE = %.3f, PSNR = %.3f, evaluation time = %.3f\n",
				math.sqrt(model2_mse / #x), model2_psnr / #x, model2_time))
      end
      if model1_am > 0 then
	 fp:write(string.format("model1  : %s = %.3f, evaluation time = %.3f\n",
				math.sqrt(model1_am / #x), model1_time))
      end
      if model2_am > 0 then
	 fp:write(string.format("model2  : %s = %.3f, evaluation time = %.3f\n",
				math.sqrt(model2_am / #x), model2_time))
      end
      fp:close()
      if detail_fp then
	 detail_fp:close()
      end
   end
   io.stdout:write("\n")
end
local function load_data_from_dir(test_dir)
   local test_x = {}
   local files = dir.getfiles(test_dir, "*.*")
   for i = 1, #files do
      local name = path.basename(files[i])
      local e = path.extension(name)
      local base = name:sub(0, name:len() - e:len())
      local img = image_loader.load_byte(files[i])
      if img then
	 table.insert(test_x, {y = iproc.crop_mod4(img),
			       basename = base})
      end
      if i % 10 == 0 then
	 if opt.show_progress then
	    xlua.progress(i, #files)
	 end
	 collectgarbage()
      end
   end
   return test_x
end
local function load_data_from_file(test_file)
   local test_x = {}
   local files = utils.split(file.read(test_file), "\n")
   for i = 1, #files do
      local name = path.basename(files[i])
      local e = path.extension(name)
      local base = name:sub(0, name:len() - e:len())
      local img = image_loader.load_byte(files[i])
      if img then
	 table.insert(test_x, {y = iproc.crop_mod4(img),
			       basename = base})
      end
      if i % 10 == 0 then
	 if opt.show_progress then
	    xlua.progress(i, #files)
	 end
	 collectgarbage()
      end
   end
   return test_x
end
local function get_basename(f)
   local name = path.basename(f)
   local e = path.extension(name)
   local base = name:sub(0, name:len() - e:len())
   return base
end
local function load_user_data(y_dir, y_file, x_dir, x_file)
   local test = {}
   local y_files
   local x_files

   if y_file:len() > 0 then
      y_files = utils.split(file.read(y_file), "\n")
   else
      y_files = dir.getfiles(y_dir, "*.*")
   end
   if x_file:len() > 0 then
      x_files = utils.split(file.read(x_file), "\n")
   else
      x_files = dir.getfiles(x_dir, "*.*")
   end
   local basename_db = {}
   for i = 1, #y_files do
      basename_db[get_basename(y_files[i])] = {y = y_files[i]}
   end
   for i = 1, #x_files do
      local key = get_basename(x_files[i])
      if basename_db[key] then
	 basename_db[key].x = x_files[i]
      else
	 error(string.format("%s is not found in %s", key, y_dir))
      end
   end
   for i = 1, #y_files do
      local key = get_basename(y_files[i])
      local d = basename_db[key]
      if not (d.x and d.y) then
	 error(string.format("%s is not found in %s", key, x_dir))
      end
   end
   for i = 1, #y_files do
      local key = get_basename(y_files[i])
      local x = image_loader.load_byte(basename_db[key].x)
      local y = image_loader.load_byte(basename_db[key].y)
      if x and y then
	 table.insert(test, {y = y,
			     x = x,
			     basename = key})
      end
      if i % 10 == 0 then
	 if opt.show_progress then
	    xlua.progress(i, #y_files)
	 end
	 collectgarbage()
      end
   end
   return test
end
function load_noise_scale_model(model_dir, noise_level, force_cudnn)
   local f = path.join(model_dir, string.format("noise%d_scale2.0x_model.t7", opt.noise_level))
   local s1, noise_scale = pcall(w2nn.load_model, f, force_cudnn)
   local model = {}
   if not s1 then
      f = path.join(model_dir, string.format("noise%d_model.t7", opt.noise_level))
      local noise
      s1, noise = pcall(w2nn.load_model, f, force_cudnn)
      if not s1 then
	 model.noise_model = nil
	 print(model_dir .. "'s noise model is not found. benchmark will use only scale model.")
      else
	 model.noise_model = noise
      end
      f = path.join(model_dir, "scale2.0x_model.t7")
      local scale
      s1, scale = pcall(w2nn.load_model, f, force_cudnn)
      if not s1 then
	 error(model_dir .. ": load error")
	 return nil
      end
      model.scale_model = scale
   else
      model.noise_scale_model = noise_scale
   end
   return model
end
if opt.show_progress then
   print(opt)
end

if opt.method == "scale" or opt.method == "scale4" then
   local f1 = path.join(opt.model1_dir, "scale2.0x_model.t7")
   local f2 = path.join(opt.model2_dir, "scale2.0x_model.t7")
   local s1, model1 = pcall(w2nn.load_model, f1, opt.force_cudnn)
   local s2, model2 = pcall(w2nn.load_model, f2, opt.force_cudnn)
   if not s1 then
      error("Load error: " .. f1)
   end
   if not s2 then
      model2 = nil
   end
   local test_x
   if opt.file:len() > 0 then
      test_x = load_data_from_file(opt.file)
   else
      test_x = load_data_from_dir(opt.dir)
   end
   benchmark(opt, test_x, model1, model2)
elseif opt.method == "noise" then
   local f1 = path.join(opt.model1_dir, string.format("noise%d_model.t7", opt.noise_level))
   local f2 = path.join(opt.model2_dir, string.format("noise%d_model.t7", opt.noise_level))
   local s1, model1 = pcall(w2nn.load_model, f1, opt.force_cudnn)
   local s2, model2 = pcall(w2nn.load_model, f2, opt.force_cudnn)
   if not s1 then
      error("Load error: " .. f1)
   end
   if not s2 then
      model2 = nil
   end
   local test_x
   if opt.file:len() > 0 then
      test_x = load_data_from_file(opt.file)
   else
      test_x = load_data_from_dir(opt.dir)
   end
   benchmark(opt, test_x, model1, model2)
elseif opt.method == "noise_scale" then
   local model2 = nil
   local model1 = load_noise_scale_model(opt.model1_dir, opt.noise_level, opt.force_cudnn)
   if opt.model2_dir:len() > 0 then
      model2 = load_noise_scale_model(opt.model2_dir, opt.noise_level, opt.force_cudnn)
   end
   local test_x
   if opt.file:len() > 0 then
      test_x = load_data_from_file(opt.file)
   else
      test_x = load_data_from_dir(opt.dir)
   end
   benchmark(opt, test_x, model1, model2)
elseif opt.method == "user" then
   local f1 = path.join(opt.model1_dir, string.format("%s_model.t7", opt.name))
   local f2 = path.join(opt.model2_dir, string.format("%s_model.t7", opt.name))
   local s1, model1 = pcall(w2nn.load_model, f1, opt.force_cudnn)
   local s2, model2 = pcall(w2nn.load_model, f2, opt.force_cudnn)
   if not s1 then
      error("Load error: " .. f1)
   end
   if not s2 then
      model2 = nil
   end
   local test = load_user_data(opt.y_dir, opt.y_file, opt.x_dir, opt.x_file)
   benchmark(opt, test, model1, model2)
elseif opt.method == "diff" then
   local test = load_user_data(opt.y_dir, opt.y_file, opt.x_dir, opt.x_file)
   benchmark(opt, test, nil, nil)
end
