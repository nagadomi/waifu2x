require 'xlua'
require 'pl'
require 'trepl'
require 'cutorch'

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
cmd:option("-seed", 11, 'RNG seed (note: it only able to reproduce the training results with `-thread 1`)')
cmd:option("-data_dir", "./data", 'path to data directory')
cmd:option("-backend", "cunn", '(cunn|cudnn)')
cmd:option("-test", "images/miku_small.png", 'path to test image')
cmd:option("-model_dir", "./models", 'model directory')
cmd:option("-method", "scale", 'method to training (noise|scale|noise_scale|user)')
cmd:option("-model", "vgg_7", 'model architecture (vgg_7|vgg_12|upconv_7|upconv_8_4x|dilated_7)')
cmd:option("-noise_level", 1, '(0|1|2|3)')
cmd:option("-style", "art", '(art|photo)')
cmd:option("-color", 'rgb', '(y|rgb)')
cmd:option("-random_color_noise_rate", 0.0, 'data augmentation using color noise (0.0-1.0)')
cmd:option("-random_overlay_rate", 0.0, 'data augmentation using flipped image overlay (0.0-1.0)')
cmd:option("-random_half_rate", 0.0, 'data augmentation using half resolution image (0.0-1.0)')
cmd:option("-random_unsharp_mask_rate", 0.0, 'data augmentation using unsharp mask (0.0-1.0)')
cmd:option("-random_blur_rate", 0.0, 'data augmentation using gaussian blur (0.0-1.0)')
cmd:option("-random_blur_size", "3,5", 'filter size for random gaussian blur (comma separated)')
cmd:option("-random_blur_sigma_min", 0.5, 'min sigma for random gaussian blur')
cmd:option("-random_blur_sigma_max", 1.0, 'max sigma for random gaussian blur')
cmd:option("-random_pairwise_scale_rate", 0.0, 'data augmentation using pairwise resize for user method')
cmd:option("-random_pairwise_scale_min", 0.85, 'min scale factor for random pairwise scale')
cmd:option("-random_pairwise_scale_max", 1.176, 'max scale factor for random pairwise scale')
cmd:option("-random_pairwise_rotate_rate", 0.0, 'data augmentation using pairwise resize for user method')
cmd:option("-random_pairwise_rotate_min", -6, 'min rotate angle for random pairwise rotate')
cmd:option("-random_pairwise_rotate_max", 6, 'max rotate angle for random pairwise rotate')
cmd:option("-random_pairwise_negate_rate", 0.0, 'data augmentation using nagate image for user method')
cmd:option("-random_pairwise_negate_x_rate", 0.0, 'data augmentation using nagate image only x side for user method')
cmd:option("-random_erasing_rate", 0.0, 'data augmentation using random erasing for user method')
cmd:option("-random_erasing_n", 1, 'number of erasing')
cmd:option("-random_erasing_rect_min", 8, 'rect min size')
cmd:option("-random_erasing_rect_max", 32, 'rect max size')
cmd:option("-pairwise_y_binary", 0, 'binarize y after data augmentation(0|1)')
cmd:option("-pairwise_flip", 1, 'use flip(0|1)')
cmd:option("-scale", 2.0, 'scale factor (2)')
cmd:option("-learning_rate", 0.00025, 'learning rate for adam')
cmd:option("-crop_size", 48, 'crop size')
cmd:option("-max_size", 256, 'if image is larger than N, image will be crop randomly')
cmd:option("-batch_size", 16, 'mini batch size')
cmd:option("-patches", 64, 'number of patch samples')
cmd:option("-inner_epoch", 4, 'number of inner epochs')
cmd:option("-epoch", 50, 'number of epochs to run')
cmd:option("-thread", -1, 'number of CPU threads')
cmd:option("-jpeg_chroma_subsampling_rate", 0.5, 'the rate of using YUV 4:2:0 in denoising training (0.0-1.0)')
cmd:option("-validation_rate", 0.05, 'validation-set rate (number_of_training_images * validation_rate > 1)')
cmd:option("-validation_crops", 200, 'number of cropping region per image in validation')
cmd:option("-active_cropping_rate", 0.5, 'active cropping rate')
cmd:option("-active_cropping_tries", 10, 'active cropping tries')
cmd:option("-nr_rate", 0.65, 'trade-off between reducing noise and erasing details (0.0-1.0)')
cmd:option("-save_history", 0, 'save all model (0|1)')
cmd:option("-plot", 0, 'plot loss chart(0|1)')
cmd:option("-downsampling_filters", "Box,Lanczos,Sinc", '(comma separated)downsampling filters for 2x scale training. (Point,Box,Triangle,Hermite,Hanning,Hamming,Blackman,Gaussian,Quadratic,Cubic,Catrom,Mitchell,Lanczos,Bessel,Sinc)')
cmd:option("-max_training_image_size", -1, 'if training image is larger than N, image will be crop randomly when data converting')
cmd:option("-use_transparent_png", 0, 'use transparent png (0|1)')
cmd:option("-resize_blur_min", 0.95, 'min blur parameter for ResizeImage')
cmd:option("-resize_blur_max", 1.05, 'max blur parameter for ResizeImage')
cmd:option("-oracle_rate", 0.1, '')
cmd:option("-oracle_drop_rate", 0.5, '')
cmd:option("-learning_rate_decay", 3.0e-7, 'learning rate decay (learning_rate * 1/(1+num_of_data*patches*epoch))')
cmd:option("-resume", "", 'resume model file')
cmd:option("-resume_epoch", 1, 'resume epoch')
cmd:option("-name", "user", 'model name for user method')
cmd:option("-gpu", "", 'GPU Device ID or ID lists (comma seprated)')
cmd:option("-loss", "huber", 'loss function (huber|l1|mse|bce)')
cmd:option("-update_criterion", "mse", 'mse|loss')
cmd:option("-padding", 0, 'replication padding size')
cmd:option("-padding_y_zero", 0, 'zero padding y for segmentation (0|1)')
cmd:option("-padding_x_zero", 0, 'zero padding x for segmentation (0|1)')
cmd:option("-grayscale", 0, 'grayscale x&y (0|1)')
cmd:option("-validation_filename_split", 0, 'make validation-set based on filename(basename)')
cmd:option("-invert_x", 0, 'invert x image in convert_lua')

local function to_bool(settings, name)
   if settings[name] == 1 then
      settings[name] = true
   else
      settings[name] = false
   end
end

local opt = cmd:parse(arg)
for k, v in pairs(opt) do
   settings[k] = v
end
to_bool(settings, "plot")
to_bool(settings, "save_history")
to_bool(settings, "use_transparent_png")
to_bool(settings, "pairwise_y_binary")
to_bool(settings, "pairwise_flip")
to_bool(settings, "padding_y_zero")
to_bool(settings, "padding_x_zero")
to_bool(settings, "grayscale")
to_bool(settings, "validation_filename_split")
to_bool(settings, "invert_x")

if settings.plot then
   require 'gnuplot'
end
if settings.save_history then
   if settings.method == "noise" then
      settings.model_file = string.format("%s/noise%d_model.%%d-%%d.t7",
					  settings.model_dir, settings.noise_level)
      settings.model_file_best = string.format("%s/noise%d_model.t7",
					       settings.model_dir, settings.noise_level)
   elseif settings.method == "scale" then
      settings.model_file = string.format("%s/scale%.1fx_model.%%d-%%d.t7",
					  settings.model_dir, settings.scale)
      settings.model_file_best = string.format("%s/scale%.1fx_model.t7",
					       settings.model_dir, settings.scale)
   elseif settings.method == "noise_scale" then
      settings.model_file = string.format("%s/noise%d_scale%.1fx_model.%%d-%%d.t7",
					  settings.model_dir,
					  settings.noise_level,
					  settings.scale)
      settings.model_file_best = string.format("%s/noise%d_scale%.1fx_model.t7",
					       settings.model_dir,
					       settings.noise_level, 
					       settings.scale)
   elseif settings.method == "user" then
      settings.model_file = string.format("%s/%s_model.%%d-%%d.t7",
					  settings.model_dir,
					  settings.name)
      settings.model_file_best = string.format("%s/%s_model.t7",
					       settings.model_dir,
					       settings.name)
   else
      error("unknown method: " .. settings.method)
   end
else
   if settings.method == "noise" then
      settings.model_file = string.format("%s/noise%d_model.t7",
					  settings.model_dir, settings.noise_level)
   elseif settings.method == "scale" then
      settings.model_file = string.format("%s/scale%.1fx_model.t7",
					  settings.model_dir, settings.scale)
   elseif settings.method == "noise_scale" then
      settings.model_file = string.format("%s/noise%d_scale%.1fx_model.t7",
					  settings.model_dir, settings.noise_level, settings.scale)
   elseif settings.method == "user" then
      settings.model_file = string.format("%s/%s_model.t7",
					  settings.model_dir, settings.name)
   else
      error("unknown method: " .. settings.method)
   end
end
if not (settings.color == "rgb" or settings.color == "y") then
   error("color must be y or rgb")
end
if not ( settings.scale == 1 or (settings.scale == math.floor(settings.scale) and settings.scale % 2 == 0)) then
   error("scale must be 1 or mod-2")
end
if not (settings.style == "art" or
	settings.style == "photo") then
   error(string.format("unknown style: %s", settings.style))
end
if settings.thread > 0 then
   torch.setnumthreads(tonumber(settings.thread))
end
if settings.downsampling_filters and settings.downsampling_filters:len() > 0 then
   settings.downsampling_filters = settings.downsampling_filters:split(",")
else
   settings.downsampling_filters = {"Box", "Lanczos", "Catrom"}
end

settings.images = string.format("%s/images.t7", settings.data_dir)
settings.image_list = string.format("%s/image_list.txt", settings.data_dir)

-- patch for lua52
if not math.log10 then
   math.log10 = function(x) return math.log(x, 10) end
end
if settings.gpu:len() > 0 then
   local gpus = {}
   local gpu_string = utils.split(settings.gpu, ",")
   for i = 1, #gpu_string do
      table.insert(gpus, tonumber(gpu_string[i]))
   end
   settings.gpu = gpus
else
   settings.gpu = {1}
end
cutorch.setDevice(settings.gpu[1])

return settings
