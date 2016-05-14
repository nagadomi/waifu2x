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
   local up = reconstruct.scale(model, settings.scale, rgb,
				settings.scale * settings.crop_size,
				settings.upsampling_filter)
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
local function make_validation_set(x, transformer, n, patches)
   n = n or 4
   local data = {}
   for i = 1, #x do
      for k = 1, math.max(n / patches, 1) do
	 local xy = transformer(x[i], true, patches)
	 for j = 1, #xy do
	    table.insert(data, {x = xy[j][1], y = xy[j][2]})
	 end
      end
      xlua.progress(i, #x)
      collectgarbage()
   end
   local new_data = {}
   local perm = torch.randperm(#data)
   for i = 1, perm:size(1) do
      new_data[i] = data[perm[i]]
   end
   data = new_data
   return data
end
local function validate(model, criterion, data, batch_size)
   local loss = 0
   local loss_count = 0
   local inputs_tmp = torch.Tensor(batch_size,
				   data[1].x:size(1), 
				   data[1].x:size(2),
				   data[1].x:size(3)):zero()
   local targets_tmp = torch.Tensor(batch_size,
				    data[1].y:size(1),
				    data[1].y:size(2),
				    data[1].y:size(3)):zero()
   local inputs = inputs_tmp:clone():cuda()
   local targets = targets_tmp:clone():cuda()
   for t = 1, #data, batch_size do
      if t + batch_size -1 > #data then
	 break
      end
      for i = 1, batch_size do
         inputs_tmp[i]:copy(data[t + i - 1].x)
	 targets_tmp[i]:copy(data[t + i - 1].y)
      end
      inputs:copy(inputs_tmp)
      targets:copy(targets_tmp)
      local z = model:forward(inputs)
      loss = loss + criterion:forward(z, targets)
      loss_count = loss_count + 1
      if loss_count % 10 == 0 then
	 xlua.progress(t, #data)
	 collectgarbage()
      end
   end
   xlua.progress(#data, #data)
   return loss / loss_count
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
      local offset = reconstruct.offset_size(model)
      local output_w = settings.crop_size - offset * 2
      local weight = torch.Tensor(1, output_w * output_w)
      weight[1]:fill(1.0)
      return w2nn.ClippedWeightedHuberCriterion(weight, 0.1, {0.0, 1.0}):cuda()
   end
end
local function transformer(model, x, is_validation, n, offset)
   x = compression.decompress(x)
   n = n or settings.patches

   if is_validation == nil then is_validation = false end
   local random_color_noise_rate = nil 
   local random_overlay_rate = nil
   local active_cropping_rate = nil
   local active_cropping_tries = nil
   if is_validation then
      active_cropping_rate = settings.active_cropping_rate
      active_cropping_tries = settings.active_cropping_tries
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
					 downsampling_filters = settings.downsampling_filters,
					 upsampling_filter = settings.upsampling_filter,
					 random_half_rate = settings.random_half_rate,
					 random_color_noise_rate = random_color_noise_rate,
					 random_overlay_rate = random_overlay_rate,
					 random_unsharp_mask_rate = settings.random_unsharp_mask_rate,
					 max_size = settings.max_size,
					 active_cropping_rate = active_cropping_rate,
					 active_cropping_tries = active_cropping_tries,
					 rgb = (settings.color == "rgb"),
					 gamma_correction = settings.gamma_correction,
					 x_upsampling = not reconstruct.has_resize(model)
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
					random_unsharp_mask_rate = settings.random_unsharp_mask_rate,
					max_size = settings.max_size,
					jpeg_chroma_subsampling_rate = settings.jpeg_chroma_subsampling_rate,
					active_cropping_rate = active_cropping_rate,
					active_cropping_tries = active_cropping_tries,
					nr_rate = settings.nr_rate,
					rgb = (settings.color == "rgb")
				     })
   end
end

local function resampling(x, y, train_x, transformer, input_size, target_size)
   print("## resampling")
   for t = 1, #train_x do
      xlua.progress(t, #train_x)
      local xy = transformer(train_x[t], false, settings.patches)
      for i = 1, #xy do
	 local index = (t - 1) * settings.patches + i
         x[index]:copy(xy[i][1])
	 y[index]:copy(xy[i][2])
      end
      if t % 50 == 0 then
	 collectgarbage()
      end
   end
end
local function remove_small_image(x)
   local new_x = {}
   for i = 1, #x do
      local x_s = compression.size(x[i])
      if x_s[2] / settings.scale > settings.crop_size + 16 and
      x_s[3] / settings.scale > settings.crop_size + 16 then
	 table.insert(new_x, x[i])
      end
      if i % 100 == 0 then
	 collectgarbage()
      end
   end
   print(string.format("removed %d small images", #x - #new_x))

   return new_x
end
local function plot(train, valid)
   gnuplot.plot({
	 {'training', torch.Tensor(train), '-'},
	 {'validation', torch.Tensor(valid), '-'}})
end
local function train()
   local hist_train = {}
   local hist_valid = {}
   local model = srcnn.create(settings.model, settings.backend, settings.color)
   local offset = reconstruct.offset_size(model)
   local pairwise_func = function(x, is_validation, n)
      return transformer(model, x, is_validation, n, offset)
   end
   local criterion = create_criterion(model)
   local eval_metric = nn.MSECriterion():cuda()
   local x = remove_small_image(torch.load(settings.images))
   local train_x, valid_x = split_data(x, math.max(math.floor(settings.validation_rate * #x), 1))
   local adam_config = {
      learningRate = settings.learning_rate,
      xBatchSize = settings.batch_size,
   }
   local lrd_count = 0
   local ch = nil
   if settings.color == "y" then
      ch = 1
   elseif settings.color == "rgb" then
      ch = 3
   end
   local best_score = 1000.0
   print("# make validation-set")
   local valid_xy = make_validation_set(valid_x, pairwise_func,
					settings.validation_crops,
					settings.patches)
   valid_x = nil
   
   collectgarbage()
   model:cuda()
   print("load .. " .. #train_x)

   local x = nil
   local y = torch.Tensor(settings.patches * #train_x,
			  ch * (settings.crop_size - offset * 2) * (settings.crop_size - offset * 2)):zero()
   if reconstruct.has_resize(model) then
      x = torch.Tensor(settings.patches * #train_x,
		       ch, settings.crop_size / settings.scale, settings.crop_size / settings.scale)
   else
      x = torch.Tensor(settings.patches * #train_x,
		       ch, settings.crop_size, settings.crop_size)
   end
   for epoch = 1, settings.epoch do
      model:training()
      print("# " .. epoch)
      resampling(x, y, train_x, pairwise_func)
      for i = 1, settings.inner_epoch do
	 local train_score = minibatch_adam(model, criterion, eval_metric, x, y, adam_config)
	 print(train_score)
	 model:evaluate()
	 print("# validation")
	 local score = validate(model, eval_metric, valid_xy, adam_config.xBatchSize)
	 table.insert(hist_train, train_score.MSE)
	 table.insert(hist_valid, score)
	 if settings.plot then
	    plot(hist_train, hist_valid)
	 end
	 if score < best_score then
	    local test_image = image_loader.load_float(settings.test) -- reload
	    lrd_count = 0
	    best_score = score
	    print("* update best model")
	    if settings.save_history then
	       torch.save(string.format(settings.model_file, epoch, i), model:clearState(), "ascii")
	       if settings.method == "noise" then
		  local log = path.join(settings.model_dir,
					("noise%d_best.%d-%d.png"):format(settings.noise_level,
									  epoch, i))
		  save_test_jpeg(model, test_image, log)
	       elseif settings.method == "scale" then
		  local log = path.join(settings.model_dir,
					("scale%.1f_best.%d-%d.png"):format(settings.scale,
									    epoch, i))
		  save_test_scale(model, test_image, log)
	       end
	    else
	       torch.save(settings.model_file, model:clearState(), "ascii")
	       if settings.method == "noise" then
		  local log = path.join(settings.model_dir,
					("noise%d_best.png"):format(settings.noise_level))
		  save_test_jpeg(model, test_image, log)
	       elseif settings.method == "scale" then
		  local log = path.join(settings.model_dir,
					("scale%.1f_best.png"):format(settings.scale))
		  save_test_scale(model, test_image, log)
	       end
	    end
	 else
	    lrd_count = lrd_count + 1
	    if lrd_count > 2 then
	       adam_config.learningRate = adam_config.learningRate * 0.8
	       print("* learning rate decay: " .. adam_config.learningRate)
	       lrd_count = 0
	    end
	 end
	 print("PSNR: " .. 10 * math.log10(1 / score) .. ", MSE: " .. score .. ", Best MSE: " .. best_score)
	 collectgarbage()
      end
   end
end
if settings.gpu > 0 then
   cutorch.setDevice(settings.gpu)
end
torch.manualSeed(settings.seed)
cutorch.manualSeed(settings.seed)
print(settings)
train()
