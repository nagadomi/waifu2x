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
   local lowres_y = nil
   local xs ={x}
   local ys = {y}
   local ls = {}

   if options.active_cropping_rate > 0 then
      lowres_y = pairwise_utils.low_resolution(y)
   end
   if options.pairwise_flip then
      xs, ys, ls = pairwise_utils.flip_augmentation(x, y, lowres_y)
   end
   assert(#xs == #ys)
   for i = 1, n do
      local t = (i % #xs) + 1
      local xc, yc = pairwise_utils.active_cropping(xs[t], ys[t], ls[t], size, scale_y,
						    options.active_cropping_rate,
						    options.active_cropping_tries)
      xc = iproc.byte2float(xc)
      yc = iproc.byte2float(yc)
      if options.rgb then
      else
	 yc = iproc.rgb2y(yc)
	 xc = iproc.rgb2y(xc)
      end
      table.insert(batch, {xc, iproc.crop(yc, offset, offset, size - offset, size - offset)})
   end

   return batch
end
return pairwise_transform
