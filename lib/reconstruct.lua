require 'image'
local iproc = require 'iproc'
local srcnn = require 'srcnn'

local function reconstruct_nn(model, x, inner_scale, offset, block_size, batch_size)
   batch_size = batch_size or 1
   if x:dim() == 2 then
      x = x:reshape(1, x:size(1), x:size(2))
   end
   local ch = x:size(1)
   local new_x = torch.Tensor(x:size(1), x:size(2) * inner_scale, x:size(3) * inner_scale):zero()
   local input_block_size = block_size
   local output_block_size = block_size * inner_scale
   local output_size = output_block_size - offset * 2
   local output_size_in_input = input_block_size - math.ceil(offset / inner_scale) * 2
   local input_indexes = {}
   local output_indexes = {}
   for i = 1, x:size(2), output_size_in_input do
      for j = 1, x:size(3), output_size_in_input do
	 if i + input_block_size - 1 <= x:size(2) and j + input_block_size - 1 <= x:size(3) then
	    local index = {{},
			   {i, i + input_block_size - 1},
			   {j, j + input_block_size - 1}}
	    local ii = (i - 1) * inner_scale + 1
	    local jj = (j - 1) * inner_scale + 1
	    local output_index = {{}, { ii , ii + output_size - 1 },
	       { jj, jj + output_size - 1}}
	    table.insert(input_indexes, index)
	    table.insert(output_indexes, output_index)
	 end
      end
   end
   local input = torch.Tensor(#input_indexes, ch, input_block_size, input_block_size)
   local input_cuda = torch.CudaTensor():resize(input:size())
   local output_cuda = torch.CudaTensor():resize(new_x:size())
   for i = 1, #input_indexes do
      input[i]:copy(x[input_indexes[i]])
      if model.w2nn_gcn then
	 local mean = input[i]:mean()
	 local stdv = input[i]:std()
	 if stdv > 0 then
	    input[i]:add(-mean):div(stdv)
	 else
	    input[i]:add(-mean)
	 end
      end
   end
   input_cuda:copy(input)
   local batch_n = math.floor(#input_indexes / batch_size)
   local batch_rem = #input_indexes % batch_size
   for i = 1, batch_n * batch_size, batch_size do
      local output = model:forward(input_cuda:narrow(1, i, batch_size))
      for j = 0, batch_size - 1 do
	 output_cuda[output_indexes[i + j]]:copy(output[j + 1])
      end
   end
   if batch_rem > 0 then
      local i = 1 + batch_n * batch_size
      local output = model:forward(input_cuda:narrow(1, i, batch_rem))
      for j = 0, batch_rem - 1 do
	 output_cuda[output_indexes[i + j]]:copy(output[j+1])
      end
   end
   new_x:copy(output_cuda)
   return new_x
end
local reconstruct = {}
function reconstruct.is_rgb(model)
   if srcnn.channels(model) == 3 then
      -- 3ch RGB
      return true
   else
      -- 1ch Y
      return false
   end
end
function reconstruct.offset_size(model)
   return srcnn.offset_size(model)
end
function reconstruct.has_resize(model)
   return srcnn.scale_factor(model) > 1
end
function reconstruct.inner_scale(model)
   return srcnn.scale_factor(model)
end
local function padding_params(x, model, block_size)
   local p = {}
   local offset = reconstruct.offset_size(model)
   p.x_w = x:size(3)
   p.x_h = x:size(2)
   p.inner_scale = reconstruct.inner_scale(model)
   local input_offset
   if model.w2nn_input_offset then
      input_offset = model.w2nn_input_offset
   else
      input_offset = math.ceil(offset / p.inner_scale)
   end
   local input_block_size = block_size
   local process_size = input_block_size - input_offset * 2
   local h_blocks = math.floor(p.x_h / process_size) +
      ((p.x_h % process_size == 0 and 0) or 1)
   local w_blocks = math.floor(p.x_w / process_size) +
      ((p.x_w % process_size == 0 and 0) or 1)
   local h = (h_blocks * process_size) + input_offset * 2
   local w = (w_blocks * process_size) + input_offset * 2
   p.pad_h1 = input_offset
   p.pad_w1 = input_offset
   p.pad_h2 = (h - input_offset) - p.x_h
   p.pad_w2 = (w - input_offset) - p.x_w
   return p
end
local function find_valid_block_size(model, block_size)
   if model.w2nn_input_size ~= nil then
      return model.w2nn_input_size
   elseif model.w2nn_valid_input_size ~= nil then
      local best_size = 0
      local best_diff = 10000
      for i = 1, #model.w2nn_valid_input_size do
	 local diff = math.abs(model.w2nn_valid_input_size[i] - block_size)
	 if diff < best_diff then
	    best_size = model.w2nn_valid_input_size[i]
	    best_diff = diff
	 end 
      end
      assert(best_size > 0)
      return best_size
   else
      return block_size
   end
end
function reconstruct.image_y(model, x, offset, block_size, batch_size)
   block_size = find_valid_block_size(model, block_size or 128)
   local p = padding_params(x, model, block_size)
   x = iproc.padding(x, p.pad_w1, p.pad_w2, p.pad_h1, p.pad_h2)
   x = x:cuda()
   x = image.rgb2yuv(x)
   local y = reconstruct_nn(model, x[1], p.inner_scale, offset, block_size, batch_size)
   x = iproc.crop(x, p.pad_w1, p.pad_h1, p.pad_w1 + p.x_w, p.pad_h1 + p.x_h)
   y = iproc.crop(y, 0, 0, p.x_w, p.x_h):clamp(0, 1)
   x[1]:copy(y)
   local output = image.yuv2rgb(x):clamp(0, 1):float()
   x = nil
   y = nil
   collectgarbage()
   return output
end
function reconstruct.scale_y(model, scale, x, offset, block_size, batch_size)
   block_size = find_valid_block_size(model, block_size or 128)
   local x_lanczos
   if reconstruct.has_resize(model) then
      x_lanczos = iproc.scale(x, x:size(3) * scale, x:size(2) * scale, "Lanczos")
   else
      x_lanczos = iproc.scale(x, x:size(3) * scale, x:size(2) * scale, "Lanczos")
      x = iproc.scale(x, x:size(3) * scale, x:size(2) * scale, "Box")
   end
   local p = padding_params(x, model, block_size)
   if p.x_w * p.x_h > 2048*2048 then
      collectgarbage()
   end
   x = iproc.padding(x, p.pad_w1, p.pad_w2, p.pad_h1, p.pad_h2)
   x = x:cuda()
   x = image.rgb2yuv(x)
   x_lanczos = image.rgb2yuv(x_lanczos)
   local y = reconstruct_nn(model, x[1], p.inner_scale, offset, block_size, batch_size)
   y = iproc.crop(y, 0, 0, p.x_w * p.inner_scale, p.x_h * p.inner_scale):clamp(0, 1)
   x_lanczos[1]:copy(y)
   local output = image.yuv2rgb(x_lanczos:cuda()):clamp(0, 1):float()
   x = nil
   x_lanczos = nil
   y = nil
   collectgarbage()
   return output
end
function reconstruct.image_rgb(model, x, offset, block_size, batch_size)
   block_size = find_valid_block_size(model, block_size or 128)
   local p = padding_params(x, model, block_size)
   x = iproc.padding(x, p.pad_w1, p.pad_w2, p.pad_h1, p.pad_h2)
   if p.x_w * p.x_h > 2048*2048 then
      collectgarbage()
   end
   local y = reconstruct_nn(model, x, p.inner_scale, offset, block_size, batch_size)
   local output = iproc.crop(y, 0, 0, p.x_w, p.x_h):clamp(0, 1)
   x = nil
   y = nil
   collectgarbage()

   return output
end
function reconstruct.scale_rgb(model, scale, x, offset, block_size, batch_size)
   block_size = find_valid_block_size(model, block_size or 128)
   if not reconstruct.has_resize(model) then
      x = iproc.scale(x, x:size(3) * scale, x:size(2) * scale, "Box")
   end
   local p = padding_params(x, model, block_size)
   x = iproc.padding(x, p.pad_w1, p.pad_w2, p.pad_h1, p.pad_h2)
   if p.x_w * p.x_h > 2048*2048 then
      collectgarbage()
   end
   local y
   y = reconstruct_nn(model, x, p.inner_scale, offset, block_size, batch_size)
   local output = iproc.crop(y, 0, 0, p.x_w * p.inner_scale, p.x_h * p.inner_scale):clamp(0, 1)
   x = nil
   y = nil
   collectgarbage()
   return output
end
function reconstruct.image(model, x, block_size, batch_size)
   local i2rgb = false
   if x:size(1) == 1 then
      local new_x = torch.Tensor(3, x:size(2), x:size(3))
      new_x[1]:copy(x)
      new_x[2]:copy(x)
      new_x[3]:copy(x)
      x = new_x
      i2rgb = true
   end
   if reconstruct.is_rgb(model) then
      x = reconstruct.image_rgb(model, x,
				reconstruct.offset_size(model), block_size, batch_size)
   else
      x = reconstruct.image_y(model, x,
			      reconstruct.offset_size(model), block_size, batch_size)
   end
   if i2rgb then
      x = image.rgb2y(x)
   end
   return x
end
function reconstruct.scale(model, scale, x, block_size, batch_size)
   local i2rgb = false
   if x:size(1) == 1 then
      local new_x = torch.Tensor(3, x:size(2), x:size(3))
      new_x[1]:copy(x)
      new_x[2]:copy(x)
      new_x[3]:copy(x)
      x = new_x
      i2rgb = true
   end
   if reconstruct.is_rgb(model) then
      x = reconstruct.scale_rgb(model, scale, x,
				reconstruct.offset_size(model),
				block_size, batch_size)
   else
      x = reconstruct.scale_y(model, scale, x,
			      reconstruct.offset_size(model),
			      block_size, batch_size)
   end
   if i2rgb then
      x = image.rgb2y(x)
   end
   return x
end
local function tr_f(a)
   return a:transpose(2, 3):contiguous() 
end
local function itr_f(a)
   return a:transpose(2, 3):contiguous()
end
local augmented_patterns = {
   {
      forward = function (a) return a end,
      backward = function (a) return a end
   },
   {
      forward = function (a) return image.hflip(a) end,
      backward = function (a) return image.hflip(a) end
   },
   {
      forward = function (a) return image.vflip(a) end,
      backward = function (a) return image.vflip(a) end
   },
   {
      forward = function (a) return image.hflip(image.vflip(a)) end,
      backward = function (a) return image.vflip(image.hflip(a)) end
   },
   {
      forward = function (a) return tr_f(a) end,
      backward = function (a) return itr_f(a) end
   },
   {
      forward = function (a) return image.hflip(tr_f(a)) end,
      backward = function (a) return itr_f(image.hflip(a)) end
   },
   {
      forward = function (a) return image.vflip(tr_f(a)) end,
      backward = function (a) return itr_f(image.vflip(a)) end
   },
   {
      forward = function (a) return image.hflip(image.vflip(tr_f(a))) end,
      backward = function (a) return itr_f(image.vflip(image.hflip(a))) end
   }
}
local function get_augmented_patterns(n)
   if n == 1 then
      -- no tta
      return {augmented_patterns[1]}
   elseif n == 2 then
      return {augmented_patterns[1], augmented_patterns[5]}
   elseif n == 4 then
      return {augmented_patterns[1], augmented_patterns[5],
	      augmented_patterns[2], augmented_patterns[7]}
   elseif n == 8 then
      return augmented_patterns
   else
      error("unsupported TTA level: " .. n)
   end
end
local function tta(f, n, model, x, block_size)
   local average = nil
   local offset = reconstruct.offset_size(model)
   local augments = get_augmented_patterns(n)
   for i = 1, #augments do 
      local out = augments[i].backward(f(model, augments[i].forward(x), offset, block_size))
      if not average then
	 average = out
      else
	 average:add(out)
      end
   end
   return average:div(#augments)
end
function reconstruct.image_tta(model, n, x, block_size)
   if model.w2nn_input_size then
      block_size = model.w2nn_input_size
   end
   if reconstruct.is_rgb(model) then
      return tta(reconstruct.image_rgb, n, model, x, block_size)
   else
      return tta(reconstruct.image_y, n, model, x, block_size)
   end
end
function reconstruct.scale_tta(model, n, scale, x, block_size)
   if model.w2nn_input_size then
      block_size = model.w2nn_input_size
   end
   if reconstruct.is_rgb(model) then
      local f = function (model, x, offset, block_size)
	 return reconstruct.scale_rgb(model, scale, x, offset, block_size)
      end
      return tta(f, n, model, x, block_size)
   else
      local f = function (model, x, offset, block_size)
	 return reconstruct.scale_y(model, scale, x, offset, block_size)
      end
      return tta(f, n, model, x, block_size)
   end
end

return reconstruct
