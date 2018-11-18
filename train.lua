require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "lib", "?.lua;") .. package.path
require 'optim'
require 'xlua'
require 'image'
require 'w2nn'
local threads = require 'threads'
local settings = require 'settings'
local srcnn = require 'srcnn'
local minibatch_adam = require 'minibatch_adam'
local iproc = require 'iproc'
local reconstruct = require 'reconstruct'
local image_loader = require 'image_loader'

local function save_test_scale(model, rgb, file)
   local up = reconstruct.scale(model, settings.scale, rgb)
   image.save(file, up)
end
local function save_test_jpeg(model, rgb, file)
   local im, count = reconstruct.image(model, rgb)
   image.save(file, im)
end
local function save_test_user(model, rgb, file)
   if settings.scale == 1 then
      save_test_jpeg(model, rgb, file)
   else
      save_test_scale(model, rgb, file)
   end
end
local function split_data(x, test_size)
   if settings.validation_filename_split then
      if not (x[1][2].data and x[1][2].data.basename) then
	 error("`images.t` does not have basename info. You need to re-run `convert_data.lua`.")
      end
      local basename_db = {}
      for i = 1, #x do
	 local meta = x[i][2].data
	 if basename_db[meta.basename] then
	    table.insert(basename_db[meta.basename], x[i])
	 else
	    basename_db[meta.basename] = {x[i]}
	 end
      end
      local basename_list = {}
      for k, v in pairs(basename_db) do
	 table.insert(basename_list, v)
      end
      local index = torch.randperm(#basename_list)
      local train_x = {}
      local valid_x = {}
      local pos = 1
      for i = 1, #basename_list do
	 if #valid_x >= test_size then
	    break
	 end
	 local xs = basename_list[index[pos]]
	 for j = 1, #xs do
	    table.insert(valid_x, xs[j])
	 end
	 pos = pos + 1
      end
      for i = pos, #basename_list do
	 local xs = basename_list[index[i]]
	 for j = 1, #xs do
	    table.insert(train_x, xs[j])
	 end
      end
      return train_x, valid_x
   else
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
end

local g_transform_pool = nil
local g_mutex = nil
local g_mutex_id = nil
local function transform_pool_init(has_resize, offset)
   local nthread = torch.getnumthreads()
   if (settings.thread > 0) then
      nthread = settings.thread
   end
   g_mutex = threads.Mutex()
   g_mutex_id = g_mutex:id()
   g_transform_pool = threads.Threads(
      nthread,
      threads.safe(
      function(threadid)
	 require 'pl'
	 local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
	 package.path = path.join(path.dirname(__FILE__), "lib", "?.lua;") .. package.path
	 require 'torch'
	 require 'nn'
	 require 'cunn'

	 torch.setnumthreads(1)
	 torch.setdefaulttensortype("torch.FloatTensor")

	 local threads = require 'threads'
	 local compression = require 'compression'
	 local pairwise_transform = require 'pairwise_transform'

	 function transformer(x, is_validation, n)
	    local mutex = threads.Mutex(g_mutex_id)
	    local meta = {data = {}}
	    local y = nil
	    if type(x) == "table" and type(x[2]) == "table" then
	       meta = x[2]
	       if x[1].x and x[1].y then
		  y = compression.decompress(x[1].y)
		  x = compression.decompress(x[1].x)
	       else
		  x = compression.decompress(x[1])
	       end
	    else
	       x = compression.decompress(x)
	    end
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
	       local conf = tablex.update({
		     mutex = mutex,
		     downsampling_filters = settings.downsampling_filters,
		     random_half_rate = settings.random_half_rate,
		     random_color_noise_rate = random_color_noise_rate,
		     random_overlay_rate = random_overlay_rate,
		     random_unsharp_mask_rate = settings.random_unsharp_mask_rate,
		     random_blur_rate = settings.random_blur_rate,
		     random_blur_size = settings.random_blur_size,
		     random_blur_sigma_min = settings.random_blur_sigma_min,
		     random_blur_sigma_max = settings.random_blur_sigma_max,
		     max_size = settings.max_size,
		     active_cropping_rate = active_cropping_rate,
		     active_cropping_tries = active_cropping_tries,
		     rgb = (settings.color == "rgb"),
		     x_upsampling = not has_resize,
		     resize_blur_min = settings.resize_blur_min,
		     resize_blur_max = settings.resize_blur_max}, meta)
	       return pairwise_transform.scale(x,
					       settings.scale,
					       settings.crop_size, offset,
					       n, conf)
	    elseif settings.method == "noise" then
	       local conf = tablex.update({
		     mutex = mutex,
		     random_half_rate = settings.random_half_rate,
		     random_color_noise_rate = random_color_noise_rate,
		     random_overlay_rate = random_overlay_rate,
		     random_unsharp_mask_rate = settings.random_unsharp_mask_rate,
		     random_blur_rate = settings.random_blur_rate,
		     random_blur_size = settings.random_blur_size,
		     random_blur_sigma_min = settings.random_blur_sigma_min,
		     random_blur_sigma_max = settings.random_blur_sigma_max,
		     max_size = settings.max_size,
		     jpeg_chroma_subsampling_rate = settings.jpeg_chroma_subsampling_rate,
		     active_cropping_rate = active_cropping_rate,
		     active_cropping_tries = active_cropping_tries,
		     nr_rate = settings.nr_rate,
		     rgb = (settings.color == "rgb")}, meta)
	       return pairwise_transform.jpeg(x,
					      settings.style,
					      settings.noise_level,
					      settings.crop_size, offset,
					      n, conf)
	    elseif settings.method == "noise_scale" then
	       local conf = tablex.update({
		     mutex = mutex,
		     downsampling_filters = settings.downsampling_filters,
		     random_half_rate = settings.random_half_rate,
		     random_color_noise_rate = random_color_noise_rate,
		     random_overlay_rate = random_overlay_rate,
		     random_unsharp_mask_rate = settings.random_unsharp_mask_rate,
		     random_blur_rate = settings.random_blur_rate,
		     random_blur_size = settings.random_blur_size,
		     random_blur_sigma_min = settings.random_blur_sigma_min,
		     random_blur_sigma_max = settings.random_blur_sigma_max,
		     max_size = settings.max_size,
		     jpeg_chroma_subsampling_rate = settings.jpeg_chroma_subsampling_rate,
		     nr_rate = settings.nr_rate,
		     active_cropping_rate = active_cropping_rate,
		     active_cropping_tries = active_cropping_tries,
		     rgb = (settings.color == "rgb"),
		     x_upsampling = not has_resize,
		     resize_blur_min = settings.resize_blur_min,
		     resize_blur_max = settings.resize_blur_max}, meta)
	       return pairwise_transform.jpeg_scale(x,
						    settings.scale,
						    settings.style,
						    settings.noise_level,
						    settings.crop_size, offset,
						    n, conf)
	    elseif settings.method == "user" then
	       local random_erasing_rate = 0
	       local random_erasing_n = 0
	       local random_erasing_rect_min = 0
	       local random_erasing_rect_max = 0
	       if is_validation then
	       else
		  random_erasing_rate = settings.random_erasing_rate
		  random_erasing_n = settings.random_erasing_n
		  random_erasing_rect_min = settings.random_erasing_rect_min
		  random_erasing_rect_max = settings.random_erasing_rect_max
	       end
	       local conf = tablex.update({
		     gcn = settings.gcn,
		     max_size = settings.max_size,
		     active_cropping_rate = active_cropping_rate,
		     active_cropping_tries = active_cropping_tries,
		     random_pairwise_rotate_rate = settings.random_pairwise_rotate_rate,
		     random_pairwise_rotate_min = settings.random_pairwise_rotate_min,
		     random_pairwise_rotate_max = settings.random_pairwise_rotate_max,
		     random_pairwise_scale_rate = settings.random_pairwise_scale_rate,
		     random_pairwise_scale_min = settings.random_pairwise_scale_min,
		     random_pairwise_scale_max = settings.random_pairwise_scale_max,
		     random_pairwise_negate_rate = settings.random_pairwise_negate_rate,
		     random_pairwise_negate_x_rate = settings.random_pairwise_negate_x_rate,
		     pairwise_y_binary = settings.pairwise_y_binary,
		     pairwise_flip = settings.pairwise_flip,
		     random_erasing_rate = random_erasing_rate,
		     random_erasing_n = random_erasing_n,
		     random_erasing_rect_min = random_erasing_rect_min,
		     random_erasing_rect_max = random_erasing_rect_max,
		     rgb = (settings.color == "rgb")}, meta)
	       return pairwise_transform.user(x, y,
					      settings.crop_size, offset,
					      n, conf)
	    end
	 end
      end)
   )
   g_transform_pool:synchronize()
end

local function make_validation_set(x, n, patches)
   local nthread = torch.getnumthreads()
   if (settings.thread > 0) then
      nthread = settings.thread
   end
   n = n or 4
   local validation_patches = math.min(16, patches or 16)
   local data = {}

   g_transform_pool:synchronize()
   torch.setnumthreads(1) -- 1

   for i = 1, #x do
      for k = 1, math.max(n / validation_patches, 1) do
	 local input = x[i]
	 g_transform_pool:addjob(
	    function()
	       local xy = transformer(input, true, validation_patches)
	       return xy
	    end,
	    function(xy)
	       for j = 1, #xy do
		  table.insert(data, {x = xy[j][1], y = xy[j][2]})
	       end
	    end
	 )
      end
      if i % 20 == 0 then
	 collectgarbage()
	 g_transform_pool:synchronize()
	 xlua.progress(i, #x)
      end
   end
   g_transform_pool:synchronize()
   torch.setnumthreads(nthread) -- revert

   local new_data = {}
   local perm = torch.randperm(#data)
   for i = 1, perm:size(1) do
      new_data[i] = data[perm[i]]
   end
   data = new_data
   return data
end
local function validate(model, criterion, eval_metric, data, batch_size)
   local psnr = 0
   local loss = 0
   local mse = 0
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
      local batch_mse = eval_metric:forward(z, targets)
      loss = loss + criterion:forward(z, targets)
      mse = mse + batch_mse
      psnr = psnr + (10 * math.log10(1 / (batch_mse + 1.0e-6)))
      loss_count = loss_count + 1
      if loss_count % 10 == 0 then
	 xlua.progress(t, #data)
	 collectgarbage()
      end
   end
   xlua.progress(#data, #data)
   return {loss = loss / loss_count, MSE = mse / loss_count, PSNR = psnr / loss_count}
end

local function create_criterion(model)
   if settings.loss == "huber" then
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
   elseif settings.loss == "l1" then
      return w2nn.L1Criterion():cuda()
   elseif settings.loss == "mse" then
      return w2nn.ClippedMSECriterion(0, 1.0):cuda()
   elseif settings.loss == "bce" then
      local bce = nn.BCECriterion()
      bce.sizeAverage = true
      return bce:cuda()
   elseif settings.loss == "aux_bce" then
      local aux = w2nn.AuxiliaryLossCriterion(nn.BCECriterion)
      aux.sizeAverage = true
      return aux:cuda()
   elseif settings.loss == "aux_huber" then
      local args = {}
      if reconstruct.is_rgb(model) then
	 local offset = reconstruct.offset_size(model)
	 local output_w = settings.crop_size - offset * 2
	 local weight = torch.Tensor(3, output_w * output_w)
	 weight[1]:fill(0.29891 * 3) -- R
	 weight[2]:fill(0.58661 * 3) -- G
	 weight[3]:fill(0.11448 * 3) -- B
	 args = {weight, 0.1, {0.0, 1.0}}
      else
	 local offset = reconstruct.offset_size(model)
	 local output_w = settings.crop_size - offset * 2
	 local weight = torch.Tensor(1, output_w * output_w)
	 weight[1]:fill(1.0)
	 args = {weight, 0.1, {0.0, 1.0}}
      end
      local aux = w2nn.AuxiliaryLossCriterion(w2nn.ClippedWeightedHuberCriterion, args)
      return aux:cuda()
   elseif settings.loss == "lbp" then
      if reconstruct.is_rgb(model) then
	 return w2nn.LBPCriterion(3, 128):cuda()
      else
	 return w2nn.LBPCriterion(1, 128):cuda()
      end
   elseif settings.loss == "lbp2" then
      if reconstruct.is_rgb(model) then
	 return w2nn.LBPCriterion(3, 128, 3, 2):cuda()
      else
	 return w2nn.LBPCriterion(1, 128, 3, 2):cuda()
      end
   elseif settings.loss == "aux_lbp" then
      if reconstruct.is_rgb(model) then
	 return w2nn.AuxiliaryLossCriterion(w2nn.LBPCriterion, {3, 128}):cuda()
      else
	 return w2nn.AuxiliaryLossCriterion(w2nn.LBPCriterion, {1, 128}):cuda()
      end
   elseif settings.loss == "aux_lbp2" then
      if reconstruct.is_rgb(model) then
	 return w2nn.AuxiliaryLossCriterion(w2nn.LBPCriterion, {3, 128, 3, 2}):cuda()
      else
	 return w2nn.AuxiliaryLossCriterion(w2nn.LBPCriterion, {1, 128, 3, 2}):cuda()
      end
   else
      error("unsupported loss .." .. settings.loss)
   end
end

local function resampling(x, y, train_x)
   local c = 1
   local shuffle = torch.randperm(#train_x)
   local nthread = torch.getnumthreads()
   if (settings.thread > 0) then
      nthread = settings.thread
   end
   torch.setnumthreads(1) -- 1

   for t = 1, #train_x do
      local input = train_x[shuffle[t]]
      g_transform_pool:addjob(
	 function()
	    local xy = transformer(input, false, settings.patches)
	    return xy
	 end,
	 function(xy)
	    for i = 1, #xy do
	       if c <= x:size(1) then
		  x[c]:copy(xy[i][1])
		  y[c]:copy(xy[i][2])
		  c = c + 1
	       else
		  break
	       end
	    end
	 end
      )
      if t % 50 == 0 then
	 collectgarbage()
	 g_transform_pool:synchronize()
	 xlua.progress(t, #train_x)
      end
      if c > x:size(1) then
	 break
      end
   end
   g_transform_pool:synchronize()
   xlua.progress(#train_x, #train_x)
   torch.setnumthreads(nthread) -- revert
end
local function get_oracle_data(x, y, instance_loss, k, samples)
   local index = torch.LongTensor(instance_loss:size(1))
   local dummy = torch.Tensor(instance_loss:size(1))
   torch.topk(dummy, index, instance_loss, k, 1, true)
   print("MSE of all data: " ..instance_loss:mean() .. ", MSE of oracle data: " .. dummy:mean())
   local shuffle = torch.randperm(k)
   local x_s = x:size()
   local y_s = y:size()
   x_s[1] = samples
   y_s[1] = samples
   local oracle_x = torch.Tensor(table.unpack(torch.totable(x_s)))
   local oracle_y = torch.Tensor(table.unpack(torch.totable(y_s)))

   for i = 1, samples do
      oracle_x[i]:copy(x[index[shuffle[i]]])
      oracle_y[i]:copy(y[index[shuffle[i]]])
   end
   return oracle_x, oracle_y
end

local function remove_small_image(x)
   local compression = require 'compression'
   local new_x = {}
   for i = 1, #x do
      local xe, meta, x_s
      xe = x[i]
      if type(x) == "table" and type(x[2]) == "table" then
	 if xe[1].x and xe[1].y then
	    x_s = compression.size(xe[1].y) -- y size
	 else
	    x_s = compression.size(xe[1])
	 end
      else
	 x_s = compression.size(xe)
      end
      if x_s[2] / settings.scale > settings.crop_size + 32 and
      x_s[3] / settings.scale > settings.crop_size + 32 then
	 table.insert(new_x, x[i])
      end
      if i % 100 == 0 then
	 collectgarbage()
      end
   end
   print(string.format("%d small images are removed", #x - #new_x))

   return new_x
end
local function plot(train, valid)
   gnuplot.plot({
	 {'training', torch.Tensor(train), '-'},
	 {'validation', torch.Tensor(valid), '-'}})
end
local function train()
   local x = torch.load(settings.images)
   if settings.method ~= "user" then
      x = remove_small_image(x)
   end
   local train_x, valid_x = split_data(x, math.max(math.floor(settings.validation_rate * #x), 1))
   local hist_train = {}
   local hist_valid = {}
   local adam_config = {
      xLearningRate = settings.learning_rate,
      xBatchSize = settings.batch_size,
      xLearningRateDecay = settings.learning_rate_decay,
      xInstanceLoss = (settings.oracle_rate > 0)
   }
   local model
   if settings.resume:len() > 0 then
      model = torch.load(settings.resume, "ascii")
      adam_config.xEvalCount = math.floor((#train_x * settings.patches) / settings.batch_size) * settings.batch_size * settings.inner_epoch * (settings.resume_epoch - 1)
      print(string.format("set eval count = %d", adam_config.xEvalCount))
      if adam_config.xEvalCount > 0 then
	 adam_config.learningRate = adam_config.xLearningRate / (1 + adam_config.xEvalCount * adam_config.xLearningRateDecay)
	 print(string.format("set learning rate = %E", adam_config.learningRate))
      else
	 adam_config.xEvalCount = 0
	 adam_config.learningRate = adam_config.xLearningRate
      end
   else
      if stringx.endswith(settings.model, ".lua") then
	 local create_model = dofile(settings.model)
	 model = create_model(srcnn, settings)
      else
	 model = srcnn.create(settings.model, settings.backend, settings.color)
      end
   end
   if model.w2nn_input_size then
      if settings.crop_size ~= model.w2nn_input_size then
	 io.stderr:write(string.format("warning: crop_size is replaced with %d\n",
				       model.w2nn_input_size))
	 settings.crop_size = model.w2nn_input_size
      end
   end
   if model.w2nn_gcn then
      settings.gcn = true
   else
      settings.gcn = false
   end
   dir.makepath(settings.model_dir)

   local offset = reconstruct.offset_size(model)
   transform_pool_init(reconstruct.has_resize(model), offset)

   local criterion = create_criterion(model)
   local eval_metric = nil
   if settings.loss:find("aux_") ~= nil then
      eval_metric = w2nn.AuxiliaryLossCriterion(w2nn.ClippedMSECriterion):cuda()
   else
      eval_metric = w2nn.ClippedMSECriterion():cuda()
   end
   local ch = nil
   if settings.color == "y" then
      ch = 1
   elseif settings.color == "rgb" then
      ch = 3
   end
   local best_score = 1000.0
   print("# make validation-set")
   local valid_xy = make_validation_set(valid_x, 
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
   local instance_loss = nil
   local pmodel = w2nn.data_parallel(model, settings.gpu)
   for epoch = settings.resume_epoch, settings.epoch do
      pmodel:training()
      print("# " .. epoch)
      if adam_config.learningRate then
	 print("learning rate: " .. adam_config.learningRate)
      end
      print("## resampling")
      if instance_loss then
	 -- active learning
	 local oracle_k = math.min(x:size(1) * (settings.oracle_rate * (1 / (1 - settings.oracle_drop_rate))), x:size(1))
	 local oracle_n = math.min(x:size(1) * settings.oracle_rate, x:size(1))
	 if oracle_n > 0 then
	    local oracle_x, oracle_y = get_oracle_data(x, y, instance_loss, oracle_k, oracle_n)
	    resampling(x:narrow(1, oracle_x:size(1) + 1, x:size(1)-oracle_x:size(1)),
		       y:narrow(1, oracle_x:size(1) + 1, x:size(1) - oracle_x:size(1)), train_x)
	    x:narrow(1, 1, oracle_x:size(1)):copy(oracle_x)
	    y:narrow(1, 1, oracle_y:size(1)):copy(oracle_y)

	    local draw_n = math.floor(math.sqrt(oracle_x:size(1), 0.5))
	    if draw_n > 100 then
	       draw_n = 100
	    end
	    image.save(path.join(settings.model_dir, "oracle_x.png"), 
		       image.toDisplayTensor({
			     input = oracle_x:narrow(1, 1, draw_n * draw_n),
			     padding = 2,
			     nrow = draw_n,
			     min = 0,
			     max = 1}))
	 else
	    resampling(x, y, train_x)
	 end
      else
	 resampling(x, y, train_x, pairwise_func)
      end
      collectgarbage()
      instance_loss = torch.Tensor(x:size(1)):zero()

      for i = 1, settings.inner_epoch do
	 pmodel:training()
	 local train_score, il = minibatch_adam(pmodel, criterion, eval_metric, x, y, adam_config)
	 instance_loss:copy(il)
	 print(train_score)
	 pmodel:evaluate()
	 print("# validation")
	 local score = validate(pmodel, criterion, eval_metric, valid_xy, adam_config.xBatchSize)
	 table.insert(hist_train, train_score.loss)
	 table.insert(hist_valid, score.loss)
	 if settings.plot then
	    plot(hist_train, hist_valid)
	 end
	 local score_for_update
	 if settings.update_criterion == "mse" then
	    score_for_update = score.MSE
	 else
	    score_for_update = score.loss
	 end
	 if score_for_update < best_score then
	    local test_image = image_loader.load_float(settings.test) -- reload
	    best_score = score_for_update
	    print("* model has updated")
	    if settings.save_history then
	       pmodel:clearState()
	       torch.save(settings.model_file_best, model, "ascii")
	       torch.save(string.format(settings.model_file, epoch, i), model, "ascii")
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
	       elseif settings.method == "noise_scale" then
		  local log = path.join(settings.model_dir,
					("noise%d_scale%.1f_best.%d-%d.png"):format(settings.noise_level, 
										    settings.scale,
										    epoch, i))
		  save_test_scale(model, test_image, log)
	       elseif settings.method == "user" then
		  local log = path.join(settings.model_dir,
					("%s_best.%d-%d.png"):format(settings.name, 
								     epoch, i))
		  save_test_user(model, test_image, log)
	       end
	    else
	       pmodel:clearState()
	       torch.save(settings.model_file, model, "ascii")
	       if settings.method == "noise" then
		  local log = path.join(settings.model_dir,
					("noise%d_best.png"):format(settings.noise_level))
		  save_test_jpeg(model, test_image, log)
	       elseif settings.method == "scale" then
		  local log = path.join(settings.model_dir,
					("scale%.1f_best.png"):format(settings.scale))
		  save_test_scale(model, test_image, log)
	       elseif settings.method == "noise_scale" then
		  local log = path.join(settings.model_dir,
					("noise%d_scale%.1f_best.png"):format(settings.noise_level, 
									      settings.scale))
		  save_test_scale(model, test_image, log)
	       elseif settings.method == "user" then
		  local log = path.join(settings.model_dir,
					("%s_best.png"):format(settings.name))
		  save_test_user(model, test_image, log)
	       end
	    end
	 end
	 print("Batch-wise PSNR: " .. score.PSNR .. ", loss: " .. score.loss .. ", MSE: " .. score.MSE .. ", best: " .. best_score)
	 collectgarbage()
      end
   end
end
torch.manualSeed(settings.seed)
cutorch.manualSeed(settings.seed)
print(settings)
train()
