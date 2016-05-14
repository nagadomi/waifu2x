require 'image'
local iproc = require 'iproc'
local srcnn = require 'srcnn'

local function reconstruct_nn(model, x, inner_scale, offset, block_size)
   if x:dim() == 2 then
      x = x:reshape(1, x:size(1), x:size(2))
   end
   local ch = x:size(1)
   local new_x = torch.Tensor(x:size(1), x:size(2) * inner_scale, x:size(3) * inner_scale):zero()
   local input_block_size = block_size / inner_scale
   local output_block_size = block_size
   local output_size = output_block_size - offset * 2
   local output_size_in_input = input_block_size - math.ceil(offset / inner_scale) * 2
   local input = torch.CudaTensor(1, ch, input_block_size, input_block_size)
   for i = 1, x:size(2), output_size_in_input do
      for j = 1, x:size(3), output_size_in_input do
	 if i + input_block_size - 1 <= x:size(2) and j + input_block_size - 1 <= x:size(3) then
	    local index = {{},
			   {i, i + input_block_size - 1},
			   {j, j + input_block_size - 1}}
	    input:copy(x[index])
	    local output = model:forward(input)
	    output = output:view(ch, output_size, output_size)
	    local ii = (i - 1) * inner_scale + 1
	    local jj = (j - 1) * inner_scale + 1
	    local output_index = {{}, { ii , ii + output_size - 1 },
	       { jj, jj + output_size - 1}}
	    new_x[output_index]:copy(output)
	 end
      end
   end
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
   local input_offset = math.ceil(offset / p.inner_scale)
   local input_block_size = block_size / p.inner_scale
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
function reconstruct.image_y(model, x, offset, block_size)
   block_size = block_size or 128
   local p = padding_params(x, model, block_size)
   x = image.rgb2yuv(iproc.padding(x, p.pad_w1, p.pad_w2, p.pad_h1, p.pad_h2))
   local y = reconstruct_nn(model, x[1], p.inner_scale, offset, block_size)
   x = iproc.crop(x, p.pad_w1, p.pad_w2, p.pad_w1 + p.x_w, p.pad_w2 + p.x_h)
   y = iproc.crop(y, 0, 0, p.x_w, p.x_h)
   y[torch.lt(y, 0)] = 0
   y[torch.gt(y, 1)] = 1
   x[1]:copy(y)
   local output = image.yuv2rgb(x)
   output[torch.lt(output, 0)] = 0
   output[torch.gt(output, 1)] = 1
   x = nil
   y = nil
   collectgarbage()
   
   return output
end
function reconstruct.scale_y(model, scale, x, offset, block_size, upsampling_filter)
   upsampling_filter = upsampling_filter or "Box"
   block_size = block_size or 128
   local x_lanczos
   if reconstruct.has_resize(model) then
      x_lanczos = x:clone()
   else
      x_lanczos = iproc.scale(x, x:size(3) * scale, x:size(2) * scale, "Lanczos")
      x = iproc.scale(x, x:size(3) * scale, x:size(2) * scale, upsampling_filter)
   end
   local p = padding_params(x, model, block_size)
   if p.x_w * p.x_h > 2048*2048 then
      collectgarbage()
   end
   x = image.rgb2yuv(iproc.padding(x, p.pad_w1, p.pad_w2, p.pad_h1, p.pad_h2))
   x_lanczos = image.rgb2yuv(x_lanczos)
   local y = reconstruct_nn(model, x[1], p.inner_scale, offset, block_size)
   y = iproc.crop(y, 0, 0, p.x_w * p.inner_scale, p.x_h * p.inner_scale)
   y[torch.lt(y, 0)] = 0
   y[torch.gt(y, 1)] = 1
   x_lanczos[1]:copy(y)
   local output = image.yuv2rgb(x_lanczos)
   output[torch.lt(output, 0)] = 0
   output[torch.gt(output, 1)] = 1
   x = nil
   x_lanczos = nil
   y = nil
   collectgarbage()
   
   return output
end
function reconstruct.image_rgb(model, x, offset, block_size)
   block_size = block_size or 128
   local p = padding_params(x, model, block_size)
   x = iproc.padding(x, p.pad_w1, p.pad_w2, p.pad_h1, p.pad_h2)
   if p.x_w * p.x_h > 2048*2048 then
      collectgarbage()
   end
   local y = reconstruct_nn(model, x, p.inner_scale, offset, block_size)
   local output = iproc.crop(y, 0, 0, p.x_w, p.x_h)
   output[torch.lt(output, 0)] = 0
   output[torch.gt(output, 1)] = 1
   x = nil
   y = nil
   collectgarbage()

   return output
end
function reconstruct.scale_rgb(model, scale, x, offset, block_size, upsampling_filter)
   upsampling_filter = upsampling_filter or "Box"
   block_size = block_size or 128
   if not reconstruct.has_resize(model) then
      x = iproc.scale(x, x:size(3) * scale, x:size(2) * scale, upsampling_filter)
   end
   local p = padding_params(x, model, block_size)
   x = iproc.padding(x, p.pad_w1, p.pad_w2, p.pad_h1, p.pad_h2)
   if p.x_w * p.x_h > 2048*2048 then
      collectgarbage()
   end
   local y
   y = reconstruct_nn(model, x, p.inner_scale, offset, block_size)
   local output = iproc.crop(y, 0, 0, p.x_w * p.inner_scale, p.x_h * p.inner_scale)
   output[torch.lt(output, 0)] = 0
   output[torch.gt(output, 1)] = 1
   x = nil
   y = nil
   collectgarbage()

   return output
end
function reconstruct.image(model, x, block_size)
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
				reconstruct.offset_size(model), block_size)
   else
      x = reconstruct.image_y(model, x,
			      reconstruct.offset_size(model), block_size)
   end
   if i2rgb then
      x = image.rgb2y(x)
   end
   return x
end
function reconstruct.scale(model, scale, x, block_size, upsampling_filter)
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
				block_size,
				upsampling_filter)
   else
      x = reconstruct.scale_y(model, scale, x,
			      reconstruct.offset_size(model),
			      block_size,
			      upsampling_filter)
   end
   if i2rgb then
      x = image.rgb2y(x)
   end
   return x
end
local function tta(f, model, x, block_size)
   local average = nil
   local offset = reconstruct.offset_size(model)
   for i = 1, 4 do 
      local flip_f, iflip_f
      if i == 1 then
	 flip_f = function (a) return a end
	 iflip_f = function (a) return a end
      elseif i == 2 then
	 flip_f = image.vflip
	 iflip_f = image.vflip
      elseif i == 3 then
	 flip_f = image.hflip
	 iflip_f = image.hflip
      elseif i == 4 then
	 flip_f = function (a) return image.hflip(image.vflip(a)) end
	 iflip_f = function (a) return image.vflip(image.hflip(a)) end
      end
      for j = 1, 2 do
	 local tr_f, itr_f
	 if j == 1 then
	    tr_f = function (a) return a end
	    itr_f = function (a) return a end
	 elseif j == 2 then
	    tr_f = function(a) return a:transpose(2, 3):contiguous() end
	    itr_f = function(a) return a:transpose(2, 3):contiguous() end
	 end
	 local out = itr_f(iflip_f(f(model, flip_f(tr_f(x)),
				     offset, block_size)))
	 if not average then
	    average = out
	 else
	    average:add(out)
	 end
      end
   end
   return average:div(8.0)
end
function reconstruct.image_tta(model, x, block_size)
   if reconstruct.is_rgb(model) then
      return tta(reconstruct.image_rgb, model, x, block_size)
   else
      return tta(reconstruct.image_y, model, x, block_size)
   end
end
function reconstruct.scale_tta(model, scale, x, block_size, upsampling_filter)
   if reconstruct.is_rgb(model) then
      local f = function (model, x, offset, block_size)
	 return reconstruct.scale_rgb(model, scale, x, offset, block_size, upsampling_filter)
      end
      return tta(f, model, x, block_size)
		 
   else
      local f = function (model, x, offset, block_size)
	 return reconstruct.scale_y(model, scale, x, offset, block_size, upsampling_filter)
      end
      return tta(f, model, x, block_size)
   end
end

return reconstruct
