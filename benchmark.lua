require './lib/portable'
require './lib/mynn'
require 'xlua'
require 'pl'

local iproc = require './lib/iproc'
local reconstruct = require './lib/reconstruct'
local image_loader = require './lib/image_loader'
local gm = require 'graphicsmagick'

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x-benchmark")
cmd:text("Options:")

cmd:option("-seed", 11, 'fixed input seed')
cmd:option("-test_dir", "./test", 'test image directory')
cmd:option("-jpeg_quality", 50, 'jpeg quality')
cmd:option("-jpeg_times", 3, 'number of jpeg compression ')
cmd:option("-jpeg_quality_down", 5, 'reducing jpeg quality each times')
cmd:option("-core", 4, 'threads')

local opt = cmd:parse(arg)
torch.setnumthreads(opt.core)
torch.setdefaulttensortype('torch.FloatTensor')

local function MSE(x1, x2)
   return (x1 - x2):pow(2):mean()
end
local function YMSE(x1, x2)
   local x1_2 = x1:clone()
   local x2_2 = x2:clone()

   x1_2[1]:mul(0.299 * 3)
   x1_2[2]:mul(0.587 * 3)
   x1_2[3]:mul(0.114 * 3)
   
   x2_2[1]:mul(0.299 * 3)
   x2_2[2]:mul(0.587 * 3)
   x2_2[3]:mul(0.114 * 3)
   
   return (x1_2 - x2_2):pow(2):mean()
end
local function PSNR(x1, x2)
   local mse = MSE(x1, x2)
   return 20 * (math.log(1.0 / math.sqrt(mse)) / math.log(10))
end
local function YPSNR(x1, x2)
   local mse = YMSE(x1, x2)
   return 20 * (math.log((0.587 * 3) / math.sqrt(mse)) / math.log(10))
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

local function noise_benchmark(x, v1_noise, v2_noise)
   local v1_mse = 0
   local v2_mse = 0
   local jpeg_mse = 0
   local v1_psnr = 0
   local v2_psnr = 0
   local jpeg_psnr = 0
   local v1_time = 0
   local v2_time = 0
   
   for i = 1, #x do
      local ground_truth = x[i]
      local jpg, blob, len, input, v1_out, v2_out, t, mse

      input = transform_jpeg(ground_truth)
      input = input:float():div(255)
      ground_truth = ground_truth:float():div(255)
      
      jpeg_mse = jpeg_mse + MSE(ground_truth, input)
      jpeg_psnr = jpeg_psnr + PSNR(ground_truth, input)
      
      t = sys.clock()
      v1_output = reconstruct.image(v1_noise, input)
      v1_time = v1_time + (sys.clock() - t)
      v1_mse = v1_mse + MSE(ground_truth, v1_output)
      v1_psnr = v1_psnr + PSNR(ground_truth, v1_output)
      
      t = sys.clock()
      v2_output = reconstruct.image(v2_noise, input)
      v2_time = v2_time + (sys.clock() - t)
      v2_mse = v2_mse + MSE(ground_truth, v2_output)
      v2_psnr = v2_psnr + PSNR(ground_truth, v2_output)
      
      io.stdout:write(
	 string.format("%d/%d; v1_time=%f, v2_time=%f, jpeg_mse=%f, v1_mse=%f, v2_mse=%f, jpeg_psnr=%f, v1_psnr=%f, v2_psnr=%f \r",
		       i, #x,
		       v1_time / i, v2_time / i,
		       jpeg_mse / i,
		       v1_mse / i, v2_mse / i,
		       jpeg_psnr / i,
		       v1_psnr / i, v2_psnr / i
	 )
      )
      io.stdout:flush()
   end
   io.stdout:write("\n")
end
local function noise_scale_benchmark(x, params, v1_noise, v1_scale, v2_noise, v2_scale)
   local v1_mse = 0
   local v2_mse = 0
   local jinc_mse = 0
   local v1_time = 0
   local v2_time = 0
   
   for i = 1, #x do
      local ground_truth = x[i]
      local downscale = iproc.scale(ground_truth,
				    ground_truth:size(3) * 0.5,
				    ground_truth:size(2) * 0.5,
				    params[i].filter)
      local jpg, blob, len, input, v1_output, v2_output, jinc_output, t, mse
      
      jpeg = gm.Image(downscale, "RGB", "DHW")
      jpeg:format("jpeg")
      blob, len = jpeg:toBlob(params[i].quality)
      jpeg:fromBlob(blob, len)
      input = jpeg:toTensor("byte", "RGB", "DHW")

      input = input:float():div(255)
      ground_truth = ground_truth:float():div(255)

      jinc_output = iproc.scale(input, input:size(3) * 2, input:size(2) * 2, "Jinc")
      jinc_mse = jinc_mse + (ground_truth - jinc_output):pow(2):mean()
      
      t = sys.clock()
      v1_output = reconstruct.image(v1_noise, input)
      v1_output = reconstruct.scale(v1_scale, 2.0, v1_output)
      v1_time = v1_time + (sys.clock() - t)
      mse = (ground_truth - v1_output):pow(2):mean()
      v1_mse = v1_mse + mse
      
      t = sys.clock()
      v2_output = reconstruct.image(v2_noise, input)
      v2_output = reconstruct.scale(v2_scale, 2.0, v2_output)
      v2_time = v2_time + (sys.clock() - t)
      mse = (ground_truth - v2_output):pow(2):mean()
      v2_mse = v2_mse + mse
      
      io.stdout:write(string.format("%d/%d; time: v1=%f, v2=%f, v1/v2=%f; mse: jinc=%f, v1=%f(%f), v2=%f(%f), v1/v2=%f \r",
				    i, #x,
				    v1_time / i, v2_time / i,
				    (v1_time / i) / (v2_time / i),
				    jinc_mse / i,
				    v1_mse / i, (v1_mse/i) / (jinc_mse/i),
				    v2_mse / i, (v2_mse/i) / (jinc_mse/i),
				    (v1_mse / i) / (v2_mse / i)))
				    
      io.stdout:flush()
   end
   io.stdout:write("\n")
end
local function scale_benchmark(x, params, v1_scale, v2_scale)
   local v1_mse = 0
   local v2_mse = 0
   local jinc_mse = 0
   local v1_psnr = 0
   local v2_psnr = 0
   local jinc_psnr = 0
   
   local v1_time = 0
   local v2_time = 0
   
   for i = 1, #x do
      local ground_truth = x[i]
      local downscale = iproc.scale(ground_truth,
				    ground_truth:size(3) * 0.5,
				    ground_truth:size(2) * 0.5,
				    params[i].filter)
      local jpg, blob, len, input, v1_output, v2_output, jinc_output, t, mse
      input = downscale

      input = input:float():div(255)
      ground_truth = ground_truth:float():div(255)

      jinc_output = iproc.scale(input, input:size(3) * 2, input:size(2) * 2, "Jinc")
      mse = (ground_truth - jinc_output):pow(2):mean()
      jinc_mse = jinc_mse + mse
      jinc_psnr = jinc_psnr + (10 * (math.log(1.0 / mse) / math.log(10)))
      
      t = sys.clock()
      v1_output = reconstruct.scale(v1_scale, 2.0, input)
      v1_time = v1_time + (sys.clock() - t)
      mse = (ground_truth - v1_output):pow(2):mean()
      v1_mse = v1_mse + mse
      v1_psnr = v1_psnr + (10 * (math.log(1.0 / mse) / math.log(10)))
      
      t = sys.clock()
      v2_output = reconstruct.scale(v2_scale, 2.0, input)
      v2_time = v2_time + (sys.clock() - t)
      mse = (ground_truth - v2_output):pow(2):mean()
      v2_mse = v2_mse + mse
      v2_psnr = v2_psnr + (10 * (math.log(1.0 / mse) / math.log(10)))
      
      io.stdout:write(string.format("%d/%d; time: v1=%f, v2=%f, v1/v2=%f; mse: jinc=%f, v1=%f(%f), v2=%f(%f), v1/v2=%f \r",
				    i, #x,
				    v1_time / i, v2_time / i,
				    (v1_time / i) / (v2_time / i),
				    jinc_psnr / i,
				    v1_psnr / i, (v1_psnr/i) / (jinc_psnr/i),
				    v2_psnr / i, (v2_psnr/i) / (jinc_psnr/i),
				    (v1_psnr / i) / (v2_psnr / i)))
				    
      io.stdout:flush()
   end
   io.stdout:write("\n")
end

local function split_data(x, test_size)
   local index = torch.randperm(#x)
   local train_size = #x - test_size
   local train_x = {}
   local valid_x = {}
   for i = 1, train_size do
      train_x[i] = x[index[i]]
   end
   for i = 1, test_size do
      valid_x[i] = x[index[train_size + i]]
   end
   return train_x, valid_x
end
local function crop_4x(x)
   local w = x:size(3) % 4
   local h = x:size(2) % 4
   return image.crop(x, 0, 0, x:size(3) - w, x:size(2) - h)
end
local function load_data(valid_dir)
   local valid_x = {}
   local files = dir.getfiles(valid_dir, "*.png")
   for i = 1, #files do
      table.insert(valid_x, crop_4x(image_loader.load_byte(files[i])))
      xlua.progress(i, #files)
   end
   return valid_x
end

local function noise_main(valid_dir, level)
   local v1_noise = torch.load(path.join(V1_DIR, string.format("noise%d_model.t7", level)), "ascii")
   local v2_noise = torch.load(path.join(V2_DIR, string.format("noise%d_model.t7", level)), "ascii")
   local valid_x = load_data(valid_dir)
   noise_benchmark(valid_x, v1_noise, v2_noise)
end
local function scale_main(valid_dir)
   local v1 = torch.load(path.join(V1_DIR, "scale2.0x_model.t7"), "ascii")
   local v2 = torch.load(path.join(V2_DIR, "scale2.0x_model.t7"), "ascii")
   local valid_x = load_data(valid_dir)
   local params = random_params(valid_x, 2)
   scale_benchmark(valid_x, params, v1, v2)
end
local function noise_scale_main(valid_dir)
   local v1_noise = torch.load(path.join(V1_DIR, "noise2_model.t7"), "ascii")
   local v1_scale = torch.load(path.join(V1_DIR, "scale2.0x_model.t7"), "ascii")
   local v2_noise = torch.load(path.join(V2_DIR, "noise2_model.t7"), "ascii")
   local v2_scale = torch.load(path.join(V2_DIR, "scale2.0x_model.t7"), "ascii")
   local valid_x = load_data(valid_dir)
   local params = random_params(valid_x, 2)
   noise_scale_benchmark(valid_x, params, v1_noise, v1_scale, v2_noise, v2_scale)
end

V1_DIR = "models/anime_style_art_rgb"
V2_DIR = "models/anime_style_art_rgb5"

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
noise_main("./test", 2)
--scale_main("./test")
--noise_scale_main("./test")
