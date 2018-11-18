local pairwise_utils = require 'pairwise_transform_utils'
local gm = {}
gm.Image = require 'graphicsmagick.Image'
local iproc = require 'iproc'
local pairwise_transform = {}

function pairwise_transform.jpeg_(src, quality, size, offset, n, options)
   local unstable_region_offset = 8
   local y = pairwise_utils.preprocess(src, size, options)
   local x = y
   local factors

   if torch.uniform() < options.jpeg_chroma_subsampling_rate then
      -- YUV 420
      factors = {2.0, 1.0, 1.0}
   else
      -- YUV 444
      factors = {1.0, 1.0, 1.0}
   end
   for i = 1, #quality do
      x = gm.Image(x, "RGB", "DHW")
      local blob, len = x:format("jpeg"):depth(8):samplingFactors(factors):toBlob(quality[i])
      x:fromBlob(blob, len)
      x = x:toTensor("byte", "RGB", "DHW")
   end
   x = iproc.crop(x, unstable_region_offset, unstable_region_offset,
		  x:size(3) - unstable_region_offset, x:size(2) - unstable_region_offset)
   y = iproc.crop(y, unstable_region_offset, unstable_region_offset,
		  y:size(3) - unstable_region_offset, y:size(2) - unstable_region_offset)
   assert(x:size(2) % 4 == 0 and x:size(3) % 4 == 0)
   assert(x:size(1) == y:size(1) and x:size(2) == y:size(2) and x:size(3) == y:size(3))
   
   local batch = {}
   local lowres_y = pairwise_utils.low_resolution(y)

   local xs, ys, ls = pairwise_utils.flip_augmentation(x, y, lowres_y)
   for i = 1, n do
      local t = (i % #xs) + 1
      local xc, yc = pairwise_utils.active_cropping(xs[t], ys[t], ls[t], size, 1,
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
      if torch.uniform() < options.nr_rate then
	 -- reducing noise
	 table.insert(batch, {xc, iproc.crop(yc, offset, offset, size - offset, size - offset)})
      else
	 -- ratain useful details
	 table.insert(batch, {yc, iproc.crop(yc, offset, offset, size - offset, size - offset)})
      end
   end
   return batch
end
function pairwise_transform.jpeg(src, style, level, size, offset, n, options)
   if style == "art" then
      if level == 0 then
	 return pairwise_transform.jpeg_(src, {torch.random(85, 95)},
					 size, offset, n, options)
      elseif level == 1 then
	 return pairwise_transform.jpeg_(src, {torch.random(65, 85)},
					 size, offset, n, options)
      elseif level == 2 or level == 3 then
	 -- level 2/3 adjusting by -nr_rate. for level3, -nr_rate=1
	 local r = torch.uniform()
	 if r > 0.4 then
	    return pairwise_transform.jpeg_(src, {torch.random(27, 70)},
					    size, offset, n, options)
	 elseif r > 0.1 then
	    local quality1 = torch.random(37, 70)
	    local quality2 = quality1 - torch.random(5, 10)
	    return pairwise_transform.jpeg_(src, {quality1, quality2},
					    size, offset, n, options)
	 else
	    local quality1 = torch.random(52, 70)
	    local quality2 = quality1 - torch.random(5, 15)
	    local quality3 = quality1 - torch.random(15, 25)
	    
	    return pairwise_transform.jpeg_(src, 
					    {quality1, quality2, quality3},
					    size, offset, n, options)
	 end
      else
	 error("unknown noise level: " .. level)
      end
   elseif style == "photo" then
      if level == 0 then
	 return pairwise_transform.jpeg_(src, {torch.random(85, 95)},
					 size, offset, n,
					 options)
      else
	 return pairwise_transform.jpeg_(src, {torch.random(37, 70)},
					 size, offset, n,
					 options)
      end
   else
      error("unknown style: " .. style)
   end
end

function pairwise_transform.test_jpeg(src)
   torch.setdefaulttensortype("torch.FloatTensor")
   local options = {random_color_noise_rate = 0.5,
		    random_half_rate = 0.5,
		    random_overlay_rate = 0.5,
		    random_unsharp_mask_rate = 0.5,
		    jpeg_chroma_subsampling_rate = 0.5,
		    nr_rate = 1.0,
		    active_cropping_rate = 0.5,
		    active_cropping_tries = 10,
		    max_size = 256,
		    rgb = true
   }
   local image = require 'image'
   local src = image.lena()
   for i = 1, 9 do
      local xy = pairwise_transform.jpeg(src,
					 "art",
					 torch.random(1, 2),
					 128, 7, 1, options)
      image.display({image = xy[1][1], legend = "y:" .. (i * 10), min=0, max=1})
      image.display({image = xy[1][2], legend = "x:" .. (i * 10), min=0, max=1})
   end
end
return pairwise_transform

