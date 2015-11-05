local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
require 'xlua'
require 'pl'

require 'w2nn'
local iproc = require 'iproc'
local reconstruct = require 'reconstruct'
local image_loader = require 'image_loader'
local gm = require 'graphicsmagick'

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x-benchmark")
cmd:text("Options:")

cmd:option("-seed", 11, 'fixed input seed')
cmd:option("-dir", "./data/test", 'test image directory')
cmd:option("-model1_dir", "./models/anime_style_art", 'model1 directory')
cmd:option("-model2_dir", "./models/anime_style_art_rgb", 'model2 directory')
cmd:option("-method", "scale", '(scale|noise)')
cmd:option("-noise_level", 1, '(1|2)')
cmd:option("-color_weight", "y", '(y|rgb)')
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

local function transform_jpeg(x)
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
local function transform_scale(x)
   return iproc.scale(x,
		      x:size(3) * 0.5,
		      x:size(2) * 0.5,
		      "Box")
end

local function benchmark(color_weight, x, input_func, v1_noise, v2_noise)
   local v1_mse = 0
   local v2_mse = 0
   local v1_psnr = 0
   local v2_psnr = 0
   
   for i = 1, #x do
      local ground_truth = x[i]
      local input, v1_output, v2_output

      input = input_func(ground_truth)
      input = input:float():div(255)
      ground_truth = ground_truth:float():div(255)
      
      t = sys.clock()
      if input:size(3) == ground_truth:size(3) then
	 v1_output = reconstruct.image(v1_noise, input)
	 v2_output = reconstruct.image(v2_noise, input)
      else
	 v1_output = reconstruct.scale(v1_noise, 2.0, input)
	 v2_output = reconstruct.scale(v2_noise, 2.0, input)
      end
      if color_weight == "y" then
	 v1_mse = v1_mse + YMSE(ground_truth, v1_output)
	 v1_psnr = v1_psnr + YPSNR(ground_truth, v1_output)
	 v2_mse = v2_mse + YMSE(ground_truth, v2_output)
	 v2_psnr = v2_psnr + YPSNR(ground_truth, v2_output)
      elseif color_weight == "rgb" then
	 v1_mse = v1_mse + MSE(ground_truth, v1_output)
	 v1_psnr = v1_psnr + PSNR(ground_truth, v1_output)
	 v2_mse = v2_mse + MSE(ground_truth, v2_output)
	 v2_psnr = v2_psnr + PSNR(ground_truth, v2_output)
      end
      
      io.stdout:write(
	 string.format("%d/%d; v1_mse=%f, v2_mse=%f, v1_psnr=%f, v2_psnr=%f \r",
		       i, #x,
		       v1_mse / i, v2_mse / i,
		       v1_psnr / i, v2_psnr / i
	 )
      )
      io.stdout:flush()
   end
   io.stdout:write("\n")
end
local function load_data(test_dir)
   local test_x = {}
   local files = dir.getfiles(test_dir, "*.*")
   for i = 1, #files do
      table.insert(test_x, iproc.crop_mod4(image_loader.load_byte(files[i])))
      xlua.progress(i, #files)
   end
   return test_x
end

print(opt)
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
if opt.method == "scale" then
   local v1 = torch.load(path.join(opt.model1_dir, "scale2.0x_model.t7"), "ascii")
   local v2 = torch.load(path.join(opt.model2_dir, "scale2.0x_model.t7"), "ascii")
   local test_x = load_data(opt.dir)
   benchmark(opt.color_weight, test_x, transform_scale, v1, v2)
elseif opt.method == "noise" then
   local v1 = torch.load(path.join(opt.model1_dir, string.format("noise%d_model.t7", opt.noise_level)), "ascii")
   local v2 = torch.load(path.join(opt.model2_dir, string.format("noise%d_model.t7", opt.noise_level)), "ascii")
   local test_x = load_data(opt.dir)
   benchmark(opt.color_weight, test_x, transform_jpeg, v1, v2)
end
