require 'image'
local gm = require 'graphicsmagick'
local iproc = require 'iproc'
local data_augmentation = require 'data_augmentation'

local pairwise_transform = {}

local function random_half(src, p)
   if torch.uniform() < p then
      local filter = ({"Box","Box","Blackman","Sinc","Lanczos", "Catrom"})[torch.random(1, 6)]
      return iproc.scale(src, src:size(3) * 0.5, src:size(2) * 0.5, filter)
   else
      return src
   end
end
local function crop_if_large(src, max_size)
   local tries = 4
   if src:size(2) > max_size and src:size(3) > max_size then
      local rect
      for i = 1, tries do
	 local yi = torch.random(0, src:size(2) - max_size)
	 local xi = torch.random(0, src:size(3) - max_size)
	 rect = iproc.crop(src, xi, yi, xi + max_size, yi + max_size)
	 -- ignore simple background
	 if rect:float():std() >= 0 then
	    break
	 end
      end
      return rect
   else
      return src
   end
end
local function preprocess(src, crop_size, options)
   local dest = src
   dest = random_half(dest, options.random_half_rate)
   dest = crop_if_large(dest, math.max(crop_size * 2, options.max_size))
   dest = data_augmentation.flip(dest)
   dest = data_augmentation.color_noise(dest, options.random_color_noise_rate)
   dest = data_augmentation.overlay(dest, options.random_overlay_rate)
   dest = data_augmentation.unsharp_mask(dest, options.random_unsharp_mask_rate)
   dest = data_augmentation.shift_1px(dest)
   
   return dest
end
local function active_cropping(x, y, size, p, tries)
   assert("x:size == y:size", x:size(2) == y:size(2) and x:size(3) == y:size(3))
   local r = torch.uniform()
   local t = "float"
   if x:type() == "torch.ByteTensor" then
      t = "byte"
   end
   if p < r then
      local xi = torch.random(0, y:size(3) - (size + 1))
      local yi = torch.random(0, y:size(2) - (size + 1))
      local xc = iproc.crop(x, xi, yi, xi + size, yi + size)
      local yc = iproc.crop(y, xi, yi, xi + size, yi + size)
      return xc, yc
   else
      local lowres = gm.Image(x, "RGB", "DHW"):
	 size(x:size(3) * 0.5, x:size(2) * 0.5, "Box"):
	 size(x:size(3), x:size(2), "Box"):
	 toTensor(t, "RGB", "DHW")
      local best_se = 0.0
      local best_xc, best_yc
      local m = torch.FloatTensor(x:size(1), size, size)
      for i = 1, tries do
	 local xi = torch.random(0, y:size(3) - (size + 1))
	 local yi = torch.random(0, y:size(2) - (size + 1))
	 local xc = iproc.crop(x, xi, yi, xi + size, yi + size)
	 local lc = iproc.crop(lowres, xi, yi, xi + size, yi + size)
	 local xcf = iproc.byte2float(xc)
	 local lcf = iproc.byte2float(lc)
	 local se = m:copy(xcf):add(-1.0, lcf):pow(2):sum()
	 if se >= best_se then
	    best_xc = xcf
	    best_yc = iproc.byte2float(iproc.crop(y, xi, yi, xi + size, yi + size))
	    best_se = se
	 end
      end
      return best_xc, best_yc
   end
end
function pairwise_transform.scale(src, scale, size, offset, n, options)
   local filters = options.downsampling_filters
   local unstable_region_offset = 8
   local downsampling_filter = filters[torch.random(1, #filters)]
   local y = preprocess(src, size, options)
   assert(y:size(2) % 4 == 0 and y:size(3) % 4 == 0)
   local down_scale = 1.0 / scale
   local x
   if options.gamma_correction then
      x = iproc.scale(iproc.scale_with_gamma22(y, y:size(3) * down_scale,
					       y:size(2) * down_scale, downsampling_filter),
		      y:size(3), y:size(2))
   else
      x = iproc.scale(iproc.scale(y, y:size(3) * down_scale,
				  y:size(2) * down_scale, downsampling_filter),
		      y:size(3), y:size(2))
   end
   x = iproc.crop(x, unstable_region_offset, unstable_region_offset,
		  x:size(3) - unstable_region_offset, x:size(2) - unstable_region_offset)
   y = iproc.crop(y, unstable_region_offset, unstable_region_offset,
		  y:size(3) - unstable_region_offset, y:size(2) - unstable_region_offset)
   assert(x:size(2) % 4 == 0 and x:size(3) % 4 == 0)
   assert(x:size(1) == y:size(1) and x:size(2) == y:size(2) and x:size(3) == y:size(3))
   
   local batch = {}
   for i = 1, n do
      local xc, yc = active_cropping(x, y,
				     size,
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
function pairwise_transform.jpeg_(src, quality, size, offset, n, options)
   local unstable_region_offset = 8
   local y = preprocess(src, size, options)
   local x = y

   for i = 1, #quality do
      x = gm.Image(x, "RGB", "DHW")
      x:format("jpeg"):depth(8)
      if torch.uniform() < options.jpeg_chroma_subsampling_rate then
	 -- YUV 420
	 x:samplingFactors({2.0, 1.0, 1.0})
      else
	 -- YUV 444
	 x:samplingFactors({1.0, 1.0, 1.0})
      end
      local blob, len = x:toBlob(quality[i])
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
   for i = 1, n do
      local xc, yc = active_cropping(x, y, size,
				     options.active_cropping_rate,
				     options.active_cropping_tries)
      xc = iproc.byte2float(xc)
      yc = iproc.byte2float(yc)
      if options.rgb then
      else
	 yc = image.rgb2yuv(yc)[1]:reshape(1, yc:size(2), yc:size(3))
	 xc = image.rgb2yuv(xc)[1]:reshape(1, xc:size(2), xc:size(3))
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
      if level == 1 then
	 return pairwise_transform.jpeg_(src, {torch.random(65, 85)},
					 size, offset, n, options)
      elseif level == 2 or level == 3 then
	 -- level 2/3 adjusting by -nr_rate. for level3, -nr_rate=1
	 local r = torch.uniform()
	 if r > 0.6 then
	    return pairwise_transform.jpeg_(src, {torch.random(27, 70)},
					    size, offset, n, options)
	 elseif r > 0.3 then
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
      -- level adjusting by -nr_rate
      return pairwise_transform.jpeg_(src, {torch.random(30, 70)},
				      size, offset, n,
				      options)
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
function pairwise_transform.test_scale(src)
   torch.setdefaulttensortype("torch.FloatTensor")
   local options = {random_color_noise_rate = 0.5,
		    random_half_rate = 0.5,
		    random_overlay_rate = 0.5,
		    random_unsharp_mask_rate = 0.5,
		    active_cropping_rate = 0.5,
		    active_cropping_tries = 10,
		    max_size = 256,
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
