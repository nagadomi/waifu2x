local pairwise_utils = require 'pairwise_transform_utils'
local iproc = require 'iproc'
local gm = require 'graphicsmagick'
local pairwise_transform = {}

local function crop_if_large(x, y, scale_y, max_size, mod)
   local tries = 4
   if y:size(2) > max_size and y:size(3) > max_size then
      assert(max_size % 4 == 0)
      local rect_x, rect_y
      for i = 1, tries do
	 local yi = torch.random(0, y:size(2) - max_size)
	 local xi = torch.random(0, y:size(3) - max_size)
	 if mod then
	    yi = yi - (yi % mod)
	    xi = xi - (xi % mod)
	 end
	 rect_y = iproc.crop(y, xi, yi, xi + max_size, yi + max_size)
	 rect_x = iproc.crop(x, xi / scale_y, yi / scale_y, xi / scale_y + max_size / scale_y, yi / scale_y + max_size / scale_y)
	 -- ignore simple background
	 if rect_y:float():std() >= 0 then
	    break
	 end
      end
      return rect_x, rect_y
   else
      return x, y
   end
end
function pairwise_transform.user(x, y, size, offset, n, options)
   assert(x:size(1) == y:size(1))

   local scale_y = y:size(2) / x:size(2)
   assert(x:size(3) == y:size(3) / scale_y)

   x, y = crop_if_large(x, y, scale_y, options.max_size, scale_y)
   assert(x:size(3) == y:size(3) / scale_y and x:size(2) == y:size(2) / scale_y)
   local batch = {}
   local lowres_y = gm.Image(y, "RGB", "DHW"):
      size(y:size(3) * 0.5, y:size(2) * 0.5, "Box"):
      size(y:size(3), y:size(2), "Box"):
      toTensor(t, "RGB", "DHW")
   local xs, ys, ls = pairwise_utils.flip_augmentation(x, y, lowres_y)
   for i = 1, n do
      local t = (i % #xs) + 1
      local xc, yc = pairwise_utils.active_cropping(xs[t], ys[t], ls[t], size, scale_y,
						    options.active_cropping_rate,
						    options.active_cropping_tries)
      xc = iproc.byte2float(xc)
      yc = iproc.byte2float(yc)
      if options.rgb then
      else
	 yc = image.rgb2yuv(yc)[1]:reshape(1, yc:size(2), yc:size(3))
	 xc = image.rgb2yuv(xc)[1]:reshape(1, xc:size(2), xc:size(3))
      end
      table.insert(batch, {xc, iproc.crop(yc, offset, offset, size - offset, size - offset)})
   end
   return batch
end
return pairwise_transform
