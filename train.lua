require './lib/portable'
require 'optim'
require 'xlua'
require 'pl'

local settings = require './lib/settings'
local minibatch_adam = require './lib/minibatch_adam'
local iproc = require './lib/iproc'
local reconstruct = require './lib/reconstruct'
local pairwise_transform = require './lib/pairwise_transform'
local image_loader = require './lib/image_loader'

local function save_test_scale(model, rgb, file)
   local up = reconstruct.scale(model, settings.scale, rgb, settings.block_offset)
   image.save(file, up)
end
local function save_test_jpeg(model, rgb, file)
   local im, count = reconstruct.image(model, rgb, settings.block_offset)
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
local function make_validation_set(x, transformer, n)
   n = n or 4
   local data = {}
   for i = 1, #x do
      for k = 1, n do
	 local x, y = transformer(x[i], true)
	 table.insert(data, {x = x:reshape(1, x:size(1), x:size(2), x:size(3)),
			     y = y:reshape(1, y:size(1), y:size(2), y:size(3))})
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
      xlua.progress(i, #data)
      if i % 10 == 0 then
	 collectgarbage()
      end
   end
   return loss / #data
end

local function train()
   local model, offset = settings.create_model(settings.color)
   assert(offset == settings.block_offset)
   local criterion = nn.MSECriterion():cuda()
   local x = torch.load(settings.images)
   local lrd_count = 0
   local train_x, valid_x = split_data(x,
				       math.floor(settings.validation_ratio * #x),
				       settings.validation_crops)
   local test = image_loader.load_float(settings.test)
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
   local transformer = function(x, is_validation)
      if is_validation == nil then is_validation = false end
      if settings.method == "scale" then
	 return pairwise_transform.scale(x,
					 settings.scale,
					 settings.crop_size, offset,
					 { color_augment = not is_validation,
					   random_half = settings.random_half,
					   rgb = (settings.color == "rgb")
					 })
      elseif settings.method == "noise" then
	 return pairwise_transform.jpeg(x,
					settings.noise_level,
					settings.crop_size, offset,
					{ color_augment = not is_validation,
					  random_half = settings.random_half,
					  rgb = (settings.color == "rgb")
					})
      elseif settings.method == "noise_scale" then
	 return pairwise_transform.jpeg_scale(x,
					      settings.scale,
					      settings.noise_level,
					      settings.crop_size, offset,
					      { color_augment = not is_validation,
						random_half = settings.random_half,
						rgb = (settings.color == "rgb")
					      })
      end
   end
   local best_score = 100000.0
   print("# make validation-set")
   local valid_xy = make_validation_set(valid_x, transformer, 20)
   valid_x = nil
   
   collectgarbage()
   model:cuda()
   print("load .. " .. #train_x)
   for epoch = 1, settings.epoch do
      model:training()
      print("# " .. epoch)
      print(minibatch_adam(model, criterion, train_x, adam_config,
			   transformer,
			   {ch, settings.crop_size, settings.crop_size},
			   {ch, settings.crop_size - offset * 2, settings.crop_size - offset * 2}
			  ))
      model:evaluate()
      print("# validation")
      local score = validate(model, criterion, valid_xy)
      if score < best_score then
	 lrd_count = 0
	 best_score = score
	 print("* update best model")
	 torch.save(settings.model_file, model)
	 if settings.method == "noise" then
	    local log = path.join(settings.model_dir,
				  ("noise%d_best.png"):format(settings.noise_level))
	    save_test_jpeg(model, test, log)
	 elseif settings.method == "scale" then
	    local log = path.join(settings.model_dir,
				  ("scale%.1f_best.png"):format(settings.scale))
	    save_test_scale(model, test, log)
	 elseif settings.method == "noise_scale" then
	    local log = path.join(settings.model_dir,
				  ("noise%d_scale%.1f_best.png"):format(settings.noise_level,
									settings.scale))
	    save_test_scale(model, test, log)
	 end
      else
	 lrd_count = lrd_count + 1
	 if lrd_count > 5 then
	    lrd_count = 0
	    adam_config.learningRate = adam_config.learningRate * 0.8
	    print("* learning rate decay: " .. adam_config.learningRate)
	 end
      end
      print("current: " .. score .. ", best: " .. best_score)
      collectgarbage()
   end
end
torch.manualSeed(settings.seed)
cutorch.manualSeed(settings.seed)
print(settings)
train()
