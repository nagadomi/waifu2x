local pairwise_utils = require 'pairwise_transform_utils'
local iproc = require 'iproc'
local gm = {}
gm.Image = require 'graphicsmagick.Image'
local pairwise_transform = {}

function pairwise_transform.user(x, y, size, offset, n, options)
   assert(x:size(1) == y:size(1))

   local scale_y = y:size(2) / x:size(2)
   assert(x:size(3) == y:size(3) / scale_y)

   x, y = pairwise_utils.preprocess_user(x, y, scale_y, size, options)
   assert(x:size(3) == y:size(3) / scale_y and x:size(2) == y:size(2) / scale_y)
   local batch = {}
   local lowres_y = pairwise_utils.low_resolution(y)
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
