require 'image'
local gm = require 'graphicsmagick'
local iproc = require 'iproc'
local data_augmentation = require 'data_augmentation'

local pairwise_transform = {}

local function random_half(src, p)
   p = p or 0.25
   --local filter = ({"Box","Blackman", "SincFast", "Jinc"})[torch.random(1, 4)]
   local filter = "Box"
   if p < torch.uniform() and (src:size(2) > 768 and src:size(3) > 1024) then
      return iproc.scale(src, src:size(3) * 0.5, src:size(2) * 0.5, filter)
   else
      return src
   end
end
local function crop_if_large(src, max_size)
   if src:size(2) > max_size and src:size(3) > max_size then
      local yi = torch.random(0, src:size(2) - max_size)
      local xi = torch.random(0, src:size(3) - max_size)
      return iproc.crop(src, xi, yi, xi + max_size, yi + max_size)
   else
      return src
   end
end
local function preprocess(src, crop_size, options)
   local dest = src
   if options.random_half then
      dest = random_half(dest)
   end
   dest = crop_if_large(dest, math.max(crop_size * 4, 512))
   dest = data_augmentation.flip(dest)
   if options.color_noise then
      dest = data_augmentation.color_noise(dest)
   end
   if options.overlay then
      dest = data_augmentation.overlay(dest)
   end
   dest = data_augmentation.shift_1px(dest)
   
   return dest
end
local function active_cropping(x, y, size, p, tries)
   assert("x:size == y:size", x:size(2) == y:size(2) and x:size(3) == y:size(3))
   local r = torch.uniform()
   if p < r then
      local xi = torch.random(0, y:size(3) - (size + 1))
      local yi = torch.random(0, y:size(2) - (size + 1))
      local xc = iproc.crop(x, xi, yi, xi + size, yi + size)
      local yc = iproc.crop(y, xi, yi, xi + size, yi + size)
      return xc, yc
   else
      local samples = {}
      local sum_mse = 0
      for i = 1, tries do
	 local xi = torch.random(0, y:size(3) - (size + 1))
	 local yi = torch.random(0, y:size(2) - (size + 1))
	 local xc = iproc.crop(x, xi, yi, xi + size, yi + size)
	 local yc = iproc.crop(y, xi, yi, xi + size, yi + size)
	 local xcf = iproc.byte2float(xc)
	 local ycf = iproc.byte2float(yc)
	 local mse = (xcf - ycf):pow(2):mean()
	 sum_mse = sum_mse + mse
	 table.insert(samples, {xc = xc, yc = yc, mse = mse})
      end
      if sum_mse > 0 then
	 table.sort(samples,
		    function (a, b)
		       return a.mse > b.mse
		    end)
      end
      return samples[1].xc, samples[1].yc
   end
end
function pairwise_transform.scale(src, scale, size, offset, n, options)
   local filters = {
      "Box","Box",  -- 0.012756949974688
      "Blackman",   -- 0.013191924552285
      --"Cartom",     -- 0.013753536746706
      --"Hanning",    -- 0.013761314529647
      --"Hermite",    -- 0.013850225205266
      "SincFast",   -- 0.014095824314306
      "Jinc",       -- 0.014244299255442
   }
   local downscale_filter = filters[torch.random(1, #filters)]
   local y = preprocess(src, size, options)
   assert(y:size(2) % 4 == 0 and y:size(3) % 4 == 0)
   local down_scale = 1.0 / scale
   local x = iproc.scale(iproc.scale(y, y:size(3) * down_scale,
				     y:size(2) * down_scale, downscale_filter),
			 y:size(3), y:size(2))
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
   local y = preprocess(src, size, options)
   local x = y
   for i = 1, #quality do
      x = gm.Image(x, "RGB", "DHW")
      x:format("jpeg")
      if options.jpeg_sampling_factors == 444 then
	 x:samplingFactors({1.0, 1.0, 1.0})
      else -- 420
	 x:samplingFactors({2.0, 1.0, 1.0})
      end
      local blob, len = x:toBlob(quality[i])
      x:fromBlob(blob, len)
      x = x:toTensor("byte", "RGB", "DHW")
   end
   -- TODO: use shift_1px after compression?
   
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
      table.insert(batch, {xc, iproc.crop(yc, offset, offset, size - offset, size - offset)})
   end
   return batch
end
function pairwise_transform.jpeg(src, category, level, size, offset, n, options)
   if category == "anime_style_art" then
      if level == 1 then
	 if torch.uniform() > 0.8 then
	    return pairwise_transform.jpeg_(src, {},
					    size, offset, n, options)
	 else
	    return pairwise_transform.jpeg_(src, {torch.random(65, 85)},
					    size, offset, n, options)
	 end
      elseif level == 2 then
	 local r = torch.uniform()
	 if torch.uniform() > 0.8 then
	    return pairwise_transform.jpeg_(src, {},
					    size, offset, n, options)
	 else
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
	 end
      else
	 error("unknown noise level: " .. level)
      end
   elseif category == "photo" then
      if level == 1 then
	 if torch.uniform() > 0.7 then
	    return pairwise_transform.jpeg_(src, {},
					    size, offset, n,
					    options)
	 else
	    return pairwise_transform.jpeg_(src, {torch.random(80, 95)},
					    size, offset, n,
					    options)
	 end
      elseif level == 2 then
	 if torch.uniform() > 0.7 then
	    return pairwise_transform.jpeg_(src, {},
					    size, offset, n,
					    options)
	 else
	    return pairwise_transform.jpeg_(src, {torch.random(65, 85)},
					    size, offset, n,
					    options)
	 end
      else
	 error("unknown noise level: " .. level)
      end
   else
      error("unknown category: " .. category)
   end
end

function pairwise_transform.test_jpeg(src)
   local options = {color_noise = true,
		    random_half = true,
		    overlay = true,
		    active_cropping_rate = 0.5,
		    active_cropping_tries = 10,
		    rgb = true
   }
   for i = 1, 9 do
      local xy = pairwise_transform.jpeg(src,
					 "anime_style_art",
					 torch.random(1, 2),
					 128, 7, 1, options)
      image.display({image = xy[1][1], legend = "y:" .. (i * 10), min=0, max=1})
      image.display({image = xy[1][2], legend = "x:" .. (i * 10), min=0, max=1})
   end
end
function pairwise_transform.test_scale(src)
   local options = {color_noise = true,
		    random_half = true,
		    overlay = true,
		    active_cropping_rate = 0.5,
		    active_cropping_tries = 10,
		    rgb = true
   }
   for i = 1, 10 do
      local xy = pairwise_transform.scale(src, 2.0, 128, 7, 1, options)
      image.display({image = xy[1][1], legend = "y:" .. (i * 10), min = 0, max = 1})
      image.display({image = xy[1][2], legend = "x:" .. (i * 10), min = 0, max = 1})
   end
end
return pairwise_transform
