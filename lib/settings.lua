require 'xlua'
require 'pl'
require 'trepl'

-- global settings

if package.preload.settings then
   return package.preload.settings
end

-- default tensor type
torch.setdefaulttensortype('torch.FloatTensor')

local settings = {}

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x-training")
cmd:text("Options:")
cmd:option("-seed", 11, 'RNG seed')
cmd:option("-data_dir", "./data", 'path to data directory')
cmd:option("-backend", "cunn", '(cunn|cudnn)')
cmd:option("-test", "images/miku_small.png", 'path to test image')
cmd:option("-model_dir", "./models", 'model directory')
cmd:option("-method", "scale", 'method to training (noise|scale)')
cmd:option("-noise_level", 1, '(1|2)')
cmd:option("-style", "art", '(art|photo)')
cmd:option("-color", 'rgb', '(y|rgb)')
cmd:option("-random_color_noise_rate", 0.0, 'data augmentation using color noise (0.0-1.0)')
cmd:option("-random_overlay_rate", 0.0, 'data augmentation using flipped image overlay (0.0-1.0)')
cmd:option("-random_half_rate", 0.0, 'data augmentation using half resolution image (0.0-1.0)')
cmd:option("-scale", 2.0, 'scale factor (2)')
cmd:option("-learning_rate", 0.00025, 'learning rate for adam')
cmd:option("-crop_size", 46, 'crop size')
cmd:option("-max_size", 256, 'if image is larger than max_size, image will be crop to max_size randomly')
cmd:option("-batch_size", 8, 'mini batch size')
cmd:option("-epoch", 200, 'number of total epochs to run')
cmd:option("-thread", -1, 'number of CPU threads')
cmd:option("-jpeg_sampling_factors", 444, '(444|420)')
cmd:option("-validation_rate", 0.05, 'validation-set rate (number_of_training_images * validation_rate > 1)')
cmd:option("-validation_crops", 80, 'number of cropping region per image in validation')
cmd:option("-active_cropping_rate", 0.5, 'active cropping rate')
cmd:option("-active_cropping_tries", 10, 'active cropping tries')
cmd:option("-nr_rate", 0.7, 'trade-off between reducing noise and erasing details (0.0-1.0)')

local opt = cmd:parse(arg)
for k, v in pairs(opt) do
   settings[k] = v
end
if settings.method == "noise" then
   settings.model_file = string.format("%s/noise%d_model.t7",
				       settings.model_dir, settings.noise_level)
elseif settings.method == "scale" then
   settings.model_file = string.format("%s/scale%.1fx_model.t7",
				       settings.model_dir, settings.scale)
elseif settings.method == "noise_scale" then
   settings.model_file = string.format("%s/noise%d_scale%.1fx_model.t7",
				       settings.model_dir, settings.noise_level, settings.scale)
else
   error("unknown method: " .. settings.method)
end
if not (settings.color == "rgb" or settings.color == "y") then
   error("color must be y or rgb")
end
if not (settings.scale == math.floor(settings.scale) and settings.scale % 2 == 0) then
   error("scale must be mod-2")
end
if not (settings.style == "art" or
	settings.style == "photo") then
   error(string.format("unknown style: %s", settings.style))
end

if settings.thread > 0 then
   torch.setnumthreads(tonumber(settings.thread))
end

settings.images = string.format("%s/images.t7", settings.data_dir)
settings.image_list = string.format("%s/image_list.txt", settings.data_dir)

return settings
