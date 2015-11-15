require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "lib", "?.lua;") .. package.path
require 'optim'
require 'xlua'

require 'w2nn'
local settings = require 'settings'
local srcnn = require 'srcnn'
local minibatch_adam = require 'minibatch_adam'
local iproc = require 'iproc'
local reconstruct = require 'reconstruct'
local compression = require 'compression'
local pairwise_transform = require 'pairwise_transform'
local image_loader = require 'image_loader'

local function save_test_scale(model, rgb, file)
   local up = reconstruct.scale(model, settings.scale, rgb)
   image.save(file, up)
end
local function save_test_jpeg(model, rgb, file)
   local im, count = reconstruct.image(model, rgb)
   image.save(file, im)
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
local function make_validation_set(x, transformer, n, batch_size)
   n = n or 4
   local data = {}
   for i = 1, #x do
      for k = 1, math.max(n / batch_size, 1) do
	 local xy = transformer(x[i], true, batch_size)
	 local tx = torch.Tensor(batch_size, xy[1][1]:size(1), xy[1][1]:size(2), xy[1][1]:size(3))
	 local ty = torch.Tensor(batch_size, xy[1][2]:size(1), xy[1][2]:size(2), xy[1][2]:size(3))
	 for j = 1, #xy do
	    tx[j]:copy(xy[j][1])
	    ty[j]:copy(xy[j][2])
	 end
	 table.insert(data, {x = tx, y = ty})
      end
      xlua.progress(i, #x)
      collectgarbage()
   end
   return data
end
local function validate(model, criterion, data)
   local loss = 0
   for i = 1, #data do
      local z = model:forward(data[i].x:cuda())
      loss = loss + criterion:forward(z, data[i].y:cuda())
      if i % 100 == 0 then
	 xlua.progress(i, #data)
	 collectgarbage()
      end
   end
   xlua.progress(#data, #data)
   return loss / #data
end

local function create_criterion(model)
   if reconstruct.is_rgb(model) then
      local offset = reconstruct.offset_size(model)
      local output_w = settings.crop_size - offset * 2
      local weight = torch.Tensor(3, output_w * output_w)
      weight[1]:fill(0.29891 * 3) -- R
      weight[2]:fill(0.58661 * 3) -- G
      weight[3]:fill(0.11448 * 3) -- B
      return w2nn.ClippedWeightedHuberCriterion(weight, 0.1, {0.0, 1.0}):cuda()
   else
      return nn.MSECriterion():cuda()
   end
end
local function transformer(x, is_validation, n, offset)
   x = compression.decompress(x)
   n = n or settings.batch_size;
   if is_validation == nil then is_validation = false end
   local random_color_noise_rate = nil 
   local random_overlay_rate = nil
   local active_cropping_rate = nil
   local active_cropping_tries = nil
   if is_validation then
      active_cropping_rate = 0
      active_cropping_tries = 0
      random_color_noise_rate = 0.0
      random_overlay_rate = 0.0
   else
      active_cropping_rate = settings.active_cropping_rate
      active_cropping_tries = settings.active_cropping_tries
      random_color_noise_rate = settings.random_color_noise_rate
      random_overlay_rate = settings.random_overlay_rate
   end
   
   if settings.method == "scale" then
      return pairwise_transform.scale(x,
				      settings.scale,
				      settings.crop_size, offset,
				      n,
				      {
					 random_half_rate = settings.random_half_rate,
					 random_color_noise_rate = random_color_noise_rate,
					 random_overlay_rate = random_overlay_rate,
					 max_size = settings.max_size,
					 active_cropping_rate = active_cropping_rate,
					 active_cropping_tries = active_cropping_tries,
					 rgb = (settings.color == "rgb")
				      })
   elseif settings.method == "noise" then
      return pairwise_transform.jpeg(x,
				     settings.style,
				     settings.noise_level,
				     settings.crop_size, offset,
				     n,
				     {
					random_half_rate = settings.random_half_rate,
					random_color_noise_rate = random_color_noise_rate,
					random_overlay_rate = random_overlay_rate,
					max_size = settings.max_size,
					jpeg_sampling_factors = settings.jpeg_sampling_factors,
					active_cropping_rate = active_cropping_rate,
					active_cropping_tries = active_cropping_tries,
					nr_rate = settings.nr_rate,
					rgb = (settings.color == "rgb")
				     })
   end
end

local function train()
   local model = srcnn.create(settings.method, settings.backend, settings.color)
   local offset = reconstruct.offset_size(model)
   local pairwise_func = function(x, is_validation, n)
      return transformer(x, is_validation, n, offset)
   end
   local criterion = create_criterion(model)
   local x = torch.load(settings.images)
   local lrd_count = 0
   local train_x, valid_x = split_data(x, math.floor(settings.validation_rate * #x))
   local adam_config = {
      learningRate = settings.learning_rate,
      xBatchSize = settings.batch_size,
   }
   local ch = nil
   if settings.color == "y" then
      ch = 1
   elseif settings.color == "rgb" then
      ch = 3
   end
   local best_score = 100000.0
   print("# make validation-set")
   local valid_xy = make_validation_set(valid_x, pairwise_func,
					settings.validation_crops,
					settings.batch_size)
   valid_x = nil
   
   collectgarbage()
   model:cuda()
   print("load .. " .. #train_x)
   for epoch = 1, settings.epoch do
      model:training()
      print("# " .. epoch)
      print(minibatch_adam(model, criterion, train_x, adam_config,
			   pairwise_func,
			   {ch, settings.crop_size, settings.crop_size},
			   {ch, settings.crop_size - offset * 2, settings.crop_size - offset * 2}
			  ))
      model:evaluate()
      print("# validation")
      local score = validate(model, criterion, valid_xy)
      if score < best_score then
	 local test_image = image_loader.load_float(settings.test) -- reload
	 lrd_count = 0
	 best_score = score
	 print("* update best model")
	 torch.save(settings.model_file, model)
	 if settings.method == "noise" then
	    local log = path.join(settings.model_dir,
				  ("noise%d_best.png"):format(settings.noise_level))
	    save_test_jpeg(model, test_image, log)
	 elseif settings.method == "scale" then
	    local log = path.join(settings.model_dir,
				  ("scale%.1f_best.png"):format(settings.scale))
	    save_test_scale(model, test_image, log)
	 end
      else
	 lrd_count = lrd_count + 1
	 if lrd_count > 5 then
	    lrd_count = 0
	    adam_config.learningRate = adam_config.learningRate * 0.9
	    print("* learning rate decay: " .. adam_config.learningRate)
	 end
      end
      print("current: " .. score .. ", best: " .. best_score)
      collectgarbage()
   end
end
if settings.gpu > 0 then
   cutorch.setDevice(settings.gpu)
end
torch.manualSeed(settings.seed)
cutorch.manualSeed(settings.seed)
print(settings)
train()
