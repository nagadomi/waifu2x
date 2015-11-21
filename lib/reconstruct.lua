require 'image'
local iproc = require 'iproc'

local function reconstruct_y(model, x, offset, block_size)
   if x:dim() == 2 then
      x = x:reshape(1, x:size(1), x:size(2))
   end
   local new_x = torch.Tensor():resizeAs(x):zero()
   local output_size = block_size - offset * 2
   local input = torch.CudaTensor(1, 1, block_size, block_size)
   
   for i = 1, x:size(2), output_size do
      for j = 1, x:size(3), output_size do
	 if i + block_size - 1 <= x:size(2) and j + block_size - 1 <= x:size(3) then
	    local index = {{},
			   {i, i + block_size - 1},
			   {j, j + block_size - 1}}
	    input:copy(x[index])
	    local output = model:forward(input):float():view(1, output_size, output_size)
	    local output_index = {{},
				  {i + offset, offset + i + output_size - 1},
				  {offset + j, offset + j + output_size - 1}}
	    new_x[output_index]:copy(output)
	 end
      end
   end
   return new_x
end
local function reconstruct_rgb(model, x, offset, block_size)
   local new_x = torch.Tensor():resizeAs(x):zero()
   local output_size = block_size - offset * 2
   local input = torch.CudaTensor(1, 3, block_size, block_size)
   
   for i = 1, x:size(2), output_size do
      for j = 1, x:size(3), output_size do
	 if i + block_size - 1 <= x:size(2) and j + block_size - 1 <= x:size(3) then
	    local index = {{},
			   {i, i + block_size - 1},
			   {j, j + block_size - 1}}
	    input:copy(x[index])
	    local output = model:forward(input):float():view(3, output_size, output_size)
	    local output_index = {{},
				  {i + offset, offset + i + output_size - 1},
				  {offset + j, offset + j + output_size - 1}}
	    new_x[output_index]:copy(output)
	 end
      end
   end
   return new_x
end
local reconstruct = {}
function reconstruct.is_rgb(model)
   if model:get(model:size() - 1).weight:size(1) == 3 then
      -- 3ch RGB
      return true
   else
      -- 1ch Y
      return false
   end
end
function reconstruct.offset_size(model)
   local conv = model:findModules("nn.SpatialConvolutionMM")
   if #conv > 0 then
      local offset = 0
      for i = 1, #conv do
	 offset = offset + (conv[i].kW - 1) / 2
      end
      return math.floor(offset)
   else
      conv = model:findModules("cudnn.SpatialConvolution")
      local offset = 0
      for i = 1, #conv do
	 offset = offset + (conv[i].kW - 1) / 2
      end
      return math.floor(offset)
   end
end
function reconstruct.image_y(model, x, offset, block_size)
   block_size = block_size or 128
   local output_size = block_size - offset * 2
   local h_blocks = math.floor(x:size(2) / output_size) +
      ((x:size(2) % output_size == 0 and 0) or 1)
   local w_blocks = math.floor(x:size(3) / output_size) +
      ((x:size(3) % output_size == 0 and 0) or 1)
   
   local h = offset + h_blocks * output_size + offset
   local w = offset + w_blocks * output_size + offset
   local pad_h1 = offset
   local pad_w1 = offset
   local pad_h2 = (h - offset) - x:size(2)
   local pad_w2 = (w - offset) - x:size(3)
   local yuv = image.rgb2yuv(iproc.padding(x, pad_w1, pad_w2, pad_h1, pad_h2))
   local y = reconstruct_y(model, yuv[1], offset, block_size)
   y[torch.lt(y, 0)] = 0
   y[torch.gt(y, 1)] = 1
   yuv[1]:copy(y)
   local output = image.yuv2rgb(iproc.crop(yuv,
					   pad_w1, pad_h1,
					   yuv:size(3) - pad_w2, yuv:size(2) - pad_h2))
   output[torch.lt(output, 0)] = 0
   output[torch.gt(output, 1)] = 1
   collectgarbage()
   
   return output
end
function reconstruct.scale_y(model, scale, x, offset, block_size)
   block_size = block_size or 128
   local x_lanczos = iproc.scale(x, x:size(3) * scale, x:size(2) * scale, "Lanczos")
   x = iproc.scale(x, x:size(3) * scale, x:size(2) * scale, "Box")

   local output_size = block_size - offset * 2
   local h_blocks = math.floor(x:size(2) / output_size) +
      ((x:size(2) % output_size == 0 and 0) or 1)
   local w_blocks = math.floor(x:size(3) / output_size) +
      ((x:size(3) % output_size == 0 and 0) or 1)
   
   local h = offset + h_blocks * output_size + offset
   local w = offset + w_blocks * output_size + offset
   local pad_h1 = offset
   local pad_w1 = offset
   local pad_h2 = (h - offset) - x:size(2)
   local pad_w2 = (w - offset) - x:size(3)
   local yuv_nn = image.rgb2yuv(iproc.padding(x, pad_w1, pad_w2, pad_h1, pad_h2))
   local yuv_lanczos = image.rgb2yuv(iproc.padding(x_lanczos, pad_w1, pad_w2, pad_h1, pad_h2))
   local y = reconstruct_y(model, yuv_nn[1], offset, block_size)
   y[torch.lt(y, 0)] = 0
   y[torch.gt(y, 1)] = 1
   yuv_lanczos[1]:copy(y)
   local output = image.yuv2rgb(iproc.crop(yuv_lanczos,
					   pad_w1, pad_h1,
					   yuv_lanczos:size(3) - pad_w2, yuv_lanczos:size(2) - pad_h2))
   output[torch.lt(output, 0)] = 0
   output[torch.gt(output, 1)] = 1
   collectgarbage()
   
   return output
end
function reconstruct.image_rgb(model, x, offset, block_size)
   block_size = block_size or 128
   local output_size = block_size - offset * 2
   local h_blocks = math.floor(x:size(2) / output_size) +
      ((x:size(2) % output_size == 0 and 0) or 1)
   local w_blocks = math.floor(x:size(3) / output_size) +
      ((x:size(3) % output_size == 0 and 0) or 1)
   
   local h = offset + h_blocks * output_size + offset
   local w = offset + w_blocks * output_size + offset
   local pad_h1 = offset
   local pad_w1 = offset
   local pad_h2 = (h - offset) - x:size(2)
   local pad_w2 = (w - offset) - x:size(3)
   local input = iproc.padding(x, pad_w1, pad_w2, pad_h1, pad_h2)
   local y = reconstruct_rgb(model, input, offset, block_size)
   local output = iproc.crop(y,
			     pad_w1, pad_h1,
			     y:size(3) - pad_w2, y:size(2) - pad_h2)
   collectgarbage()
   output[torch.lt(output, 0)] = 0
   output[torch.gt(output, 1)] = 1
   
   return output
end
function reconstruct.scale_rgb(model, scale, x, offset, block_size)
   block_size = block_size or 128
   x = iproc.scale(x, x:size(3) * scale, x:size(2) * scale, "Box")

   local output_size = block_size - offset * 2
   local h_blocks = math.floor(x:size(2) / output_size) +
      ((x:size(2) % output_size == 0 and 0) or 1)
   local w_blocks = math.floor(x:size(3) / output_size) +
      ((x:size(3) % output_size == 0 and 0) or 1)
   
   local h = offset + h_blocks * output_size + offset
   local w = offset + w_blocks * output_size + offset
   local pad_h1 = offset
   local pad_w1 = offset
   local pad_h2 = (h - offset) - x:size(2)
   local pad_w2 = (w - offset) - x:size(3)
   local input = iproc.padding(x, pad_w1, pad_w2, pad_h1, pad_h2)
   local y = reconstruct_rgb(model, input, offset, block_size)
   local output = iproc.crop(y,
			     pad_w1, pad_h1,
			     y:size(3) - pad_w2, y:size(2) - pad_h2)
   output[torch.lt(output, 0)] = 0
   output[torch.gt(output, 1)] = 1
   collectgarbage()
   
   return output
end

function reconstruct.image(model, x, block_size)
   if reconstruct.is_rgb(model) then
      return reconstruct.image_rgb(model, x,
				   reconstruct.offset_size(model), block_size)
   else
      return reconstruct.image_y(model, x,
				 reconstruct.offset_size(model), block_size)
   end
end
function reconstruct.scale(model, scale, x, block_size)
   if reconstruct.is_rgb(model) then
      return reconstruct.scale_rgb(model, scale, x,
				   reconstruct.offset_size(model), block_size)
   else
      return reconstruct.scale_y(model, scale, x,
				 reconstruct.offset_size(model), block_size)
   end
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
function reconstruct.scale_tta(model, scale, x, block_size)
   if reconstruct.is_rgb(model) then
      local f = function (model, x, offset, block_size)
	 return reconstruct.scale_rgb(model, scale, x, offset, block_size)
      end
      return tta(f, model, x, block_size)
		 
   else
      local f = function (model, x, offset, block_size)
	 return reconstruct.scale_y(model, scale, x, offset, block_size)
      end
      return tta(f, model, x, block_size)
   end
end

return reconstruct
