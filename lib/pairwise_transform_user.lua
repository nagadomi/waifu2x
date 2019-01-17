local pairwise_utils = require 'pairwise_transform_utils'
local data_augmentation = require 'data_augmentation'
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
   if options.pairwise_flip and n == 1 then
      xs[1], ys[1] = data_augmentation.pairwise_flip(xs[1], ys[1])
   elseif options.pairwise_flip then
      xs, ys, ls = pairwise_utils.flip_augmentation(x, y, lowres_y)
   end
   assert(#xs == #ys)
   local perm = torch.randperm(#xs)
   for i = 1, n do
      local t = perm[(i % #xs) + 1]
      local xc, yc = pairwise_utils.active_cropping(xs[t], ys[t], ls[t], size, scale_y,
						    options.active_cropping_rate,
						    options.active_cropping_tries)
      xc = iproc.byte2float(xc)
      yc = iproc.byte2float(yc)
      if options.rgb then
      else
	 if xc:size(1) > 1 then
	    yc = iproc.rgb2y(yc)
	    xc = iproc.rgb2y(xc)
	 end
      end
      if options.gcn then
	 local mean = xc:mean()
	 local stdv = xc:std()
	 if stdv > 0 then
	    xc:add(-mean):div(stdv)
	 else
	    xc:add(-mean)
	 end
      end
      yc = iproc.crop(yc, offset, offset, size - offset, size - offset)
      if options.pairwise_y_binary then
	 yc[torch.lt(yc, 0.5)] = 0
	 yc[torch.gt(yc, 0)] = 1
      end
      table.insert(batch, {xc, yc})
   end

   return batch
end
return pairwise_transform
