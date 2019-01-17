local pairwise_utils = require 'pairwise_transform_utils'
local iproc = require 'iproc'
local gm = {}
gm.Image = require 'graphicsmagick.Image'
local pairwise_transform = {}

function pairwise_transform.scale(src, scale, size, offset, n, options)
   local filters = options.downsampling_filters
   if options.data.filters then
      filters = options.data.filters
   end
   local unstable_region_offset = 8
   local downsampling_filter = filters[torch.random(1, #filters)]
   local blur = torch.uniform(options.resize_blur_min, options.resize_blur_max)
   local y = pairwise_utils.preprocess(src, size, options)
   assert(y:size(2) % 4 == 0 and y:size(3) % 4 == 0)
   local down_scale = 1.0 / scale
   local x
   local small = iproc.scale(y, y:size(3) * down_scale,
				  y:size(2) * down_scale, downsampling_filter, blur)
   if options.x_upsampling then
      x = iproc.scale(small, y:size(3), y:size(2), "Box")
   else
      x = small
   end
   local scale_inner = scale
   if options.x_upsampling then
      scale_inner = 1
   end
   x = iproc.crop(x, unstable_region_offset, unstable_region_offset,
		  x:size(3) - unstable_region_offset, x:size(2) - unstable_region_offset)
   y = iproc.crop(y, unstable_region_offset * scale_inner, unstable_region_offset * scale_inner,
		  y:size(3) - unstable_region_offset * scale_inner, y:size(2) - unstable_region_offset * scale_inner)
   if options.x_upsampling then
      assert(x:size(2) % 4 == 0 and x:size(3) % 4 == 0)
      assert(x:size(1) == y:size(1) and x:size(2) == y:size(2) and x:size(3) == y:size(3))
   else
      assert(x:size(1) == y:size(1) and x:size(2) * scale == y:size(2) and x:size(3) * scale == y:size(3))
   end
   local batch = {}
   local lowres_y = pairwise_utils.low_resolution(y)
   local xs, ys, ls, _ = pairwise_utils.flip_augmentation(x, y, lowres_y)
   for i = 1, n do
      local t = (i % #xs) + 1
      local xc, yc = pairwise_utils.active_cropping(xs[t], ys[t], ls[t],
						    size,
						    scale_inner,
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
      table.insert(batch, {xc, iproc.crop(yc, offset, offset, size - offset, size - offset)})
   end
   return batch
end
function pairwise_transform.test_scale(src)
   torch.setdefaulttensortype("torch.FloatTensor")
   local options = {random_color_noise_rate = 0.5,
		    random_half_rate = 0.5,
		    random_overlay_rate = 0.5,
		    random_unsharp_mask_rate = 0.5,
		    active_cropping_rate = 0.5,
		    active_cropping_tries = 10,
		    max_size = 256,
		    x_upsampling = false,
		    downsampling_filters = "Box",
		    rgb = true
   }
   local image = require 'image'
   local src = image.lena()

   for i = 1, 10 do
      local xy = pairwise_transform.scale(src, 2.0, 128, 7, 1, options)
      image.display({image = xy[1][1], legend = "y:" .. (i * 10), min = 0, max = 1})
      image.display({image = xy[1][2], legend = "x:" .. (i * 10), min = 0, max = 1})
   end
end
return pairwise_transform
