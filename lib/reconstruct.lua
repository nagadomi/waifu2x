require 'image'
local iproc = require './iproc'

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
function model_is_rgb(model)
   if model:get(model:size() - 1).weight:size(1) == 3 then
      -- 3ch RGB
      return true
   else
      -- 1ch Y
      return false
   end
end

local reconstruct = {}
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
   local output = image.yuv2rgb(image.crop(yuv,
					   pad_w1, pad_h1,
					   yuv:size(3) - pad_w2, yuv:size(2) - pad_h2))
   output[torch.lt(output, 0)] = 0
   output[torch.gt(output, 1)] = 1
   collectgarbage()
   
   return output
end
function reconstruct.scale_y(model, scale, x, offset, block_size)
   block_size = block_size or 128
   local x_jinc = iproc.scale(x, x:size(3) * scale, x:size(2) * scale, "Jinc")
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
   local yuv_jinc = image.rgb2yuv(iproc.padding(x_jinc, pad_w1, pad_w2, pad_h1, pad_h2))
   local y = reconstruct_y(model, yuv_nn[1], offset, block_size)
   y[torch.lt(y, 0)] = 0
   y[torch.gt(y, 1)] = 1
   yuv_jinc[1]:copy(y)
   local output = image.yuv2rgb(image.crop(yuv_jinc,
					   pad_w1, pad_h1,
					   yuv_jinc:size(3) - pad_w2, yuv_jinc:size(2) - pad_h2))
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
   local output = image.crop(y,
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
   local output = image.crop(y,
			     pad_w1, pad_h1,
			     y:size(3) - pad_w2, y:size(2) - pad_h2)
   output[torch.lt(output, 0)] = 0
   output[torch.gt(output, 1)] = 1
   collectgarbage()
   
   return output
end

function reconstruct.image(model, x, offset, block_size)
   if model_is_rgb(model) then
      return reconstruct.image_rgb(model, x, offset, block_size)
   else
      return reconstruct.image_y(model, x, offset, block_size)
   end
end
function reconstruct.scale(model, scale, x, offset, block_size)
   if model_is_rgb(model) then
      return reconstruct.scale_rgb(model, scale, x, offset, block_size)
   else
      return reconstruct.scale_y(model, scale, x, offset, block_size)
   end
end

return reconstruct
