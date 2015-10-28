require 'image'
local gm = require 'graphicsmagick'
local iproc = require 'iproc'
local reconstruct = require 'reconstruct'
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
local function pcacov(x)
   local mean = torch.mean(x, 1)
   local xm = x - torch.ger(torch.ones(x:size(1)), mean:squeeze())
   local c = torch.mm(xm:t(), xm)
   c:div(x:size(1) - 1)
   local ce, cv = torch.symeig(c, 'V')
   return ce, cv
end
local function crop_if_large(src, max_size)
   if src:size(2) > max_size and src:size(3) > max_size then
      local yi = torch.random(0, src:size(2) - max_size)
      local xi = torch.random(0, src:size(3) - max_size)
      return image.crop(src, xi, yi, xi + max_size, yi + max_size)
   else
      return src
   end
end
local function active_cropping(x, y, size, offset, p, tries)
   assert("x:size == y:size", x:size(2) == y:size(2) and x:size(3) == y:size(3))
   local r = torch.uniform()
   if p < r then
      local xi = torch.random(offset, y:size(3) - (size + offset + 1))
      local yi = torch.random(offset, y:size(2) - (size + offset + 1))
      local xc = image.crop(x, xi, yi, xi + size, yi + size)
      local yc = image.crop(y, xi, yi, xi + size, yi + size)
      yc = yc:float():div(255)
      xc = xc:float():div(255)
      return xc, yc
   else
      local samples = {}
      local sum_mse = 0
      for i = 1, tries do
	 local xi = torch.random(offset, y:size(3) - (size + offset + 1))
	 local yi = torch.random(offset, y:size(2) - (size + offset + 1))
	 local xc = image.crop(x, xi, yi, xi + size, yi + size):float():div(255)
	 local yc = image.crop(y, xi, yi, xi + size, yi + size):float():div(255)
	 local mse = (xc - yc):pow(2):mean()
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

local function color_noise(src)
   local p = 0.1
   src = src:float():div(255)
   local src_t = src:reshape(src:size(1), src:nElement() / src:size(1)):t():contiguous()
   local ce, cv = pcacov(src_t)
   local color_scale = torch.Tensor(3):uniform(1 / (1 + p), 1 + p)
   
   pca_space = torch.mm(src_t, cv):t():contiguous()
   for i = 1, 3 do
      pca_space[i]:mul(color_scale[i])
   end
   x = torch.mm(pca_space:t(), cv:t()):t():contiguous():resizeAs(src)
   x[torch.lt(x, 0.0)] = 0.0
   x[torch.gt(x, 1.0)] = 1.0
   
   return x:mul(255):byte()
end
local function shift_1px(src)
   -- reducing the even/odd issue in nearest neighbor.
   local r = torch.random(1, 4)
   
end
local function flip_augment(x, y)
   local flip = torch.random(1, 4)
   if y then
      if flip == 1 then
	 x = image.hflip(x)
	 y = image.hflip(y)
      elseif flip == 2 then
	 x = image.vflip(x)
	 y = image.vflip(y)
      elseif flip == 3 then
	 x = image.hflip(image.vflip(x))
	 y = image.hflip(image.vflip(y))
      elseif flip == 4 then
      end
      return x, y
   else
      if flip == 1 then
	 x = image.hflip(x)
      elseif flip == 2 then
	 x = image.vflip(x)
      elseif flip == 3 then
	 x = image.hflip(image.vflip(x))
      elseif flip == 4 then
      end
      return x
   end
end
local function overlay_augment(src, p)
   p = p or 0.25
   if torch.uniform() > (1.0 - p) then
      local r = torch.uniform(0.2, 0.8)
      local t = "float"
      if src:type() == "torch.ByteTensor" then
	 src = src:float():div(255)
	 t = "byte"
      end
      local flip = flip_augment(src)
      flip:mul(r):add(src * (1.0 - r))
      if t == "byte" then
	 flip = flip:mul(255):byte()
      end
      return flip
   else
      return src
   end
end
local function data_augment(y, options)
   y = flip_augment(y)
   if options.color_noise then
      y = color_noise(y)
   end
   if options.overlay then
      y = overlay_augment(y)
   end
   return y
end

local INTERPOLATION_PADDING = 16
function pairwise_transform.scale(src, scale, size, offset, n, options)
   local filters = {
      "Box","Box",  -- 0.012756949974688
      "Blackman",   -- 0.013191924552285
      --"Cartom",     -- 0.013753536746706
      --"Hanning",    -- 0.013761314529647
      --"Hermite",    -- 0.013850225205266
      "SincFast",   -- 0.014095824314306
      --"Jinc",       -- 0.014244299255442
   }
   if options.random_half then
      src = random_half(src)
   end
   local downscale_filter = filters[torch.random(1, #filters)]
   local y = data_augment(crop_if_large(src, math.max(size * 4, 512)), options)
   local down_scale = 1.0 / scale
   local x = iproc.scale(iproc.scale(y, y:size(3) * down_scale,
				     y:size(2) * down_scale, downscale_filter),
			 y:size(3), y:size(2))
   local batch = {}
   for i = 1, n do
      local xc, yc = active_cropping(x, y,
				     size,
				     INTERPOLATION_PADDING,
				     options.active_cropping_rate,
				     options.active_cropping_tries)
      if options.rgb then
      else
	 yc = image.rgb2yuv(yc)[1]:reshape(1, yc:size(2), yc:size(3))
	 xc = image.rgb2yuv(xc)[1]:reshape(1, xc:size(2), xc:size(3))
      end
      table.insert(batch, {xc, image.crop(yc, offset, offset, size - offset, size - offset)})
   end
   return batch
end
function pairwise_transform.jpeg_(src, quality, size, offset, n, options)
   local y = data_augment(crop_if_large(src, math.max(size * 4, 512)), options)   
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
   
   local batch = {}
   for i = 1, n do
      local xc, yc = active_cropping(x, y, size, 0,
				     options.active_cropping_rate,
				     options.active_cropping_tries)
      xc, yc = flip_augment(xc, yc)
      
      if options.rgb then
      else
	 yc = image.rgb2yuv(yc)[1]:reshape(1, yc:size(2), yc:size(3))
	 xc = image.rgb2yuv(xc)[1]:reshape(1, xc:size(2), xc:size(3))
      end
      table.insert(batch, {xc, image.crop(yc, offset, offset, size - offset, size - offset)})
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
function pairwise_transform.jpeg_scale_(src, scale, quality, size, offset, options)
   if options.random_half then
      src = random_half(src)
   end
   src = crop_if_large(src, math.max(size * 4, 512))
   local down_scale = 1.0 / scale
   local filters = {
      "Box",        -- 0.012756949974688
      "Blackman",   -- 0.013191924552285
      --"Cartom",     -- 0.013753536746706
      --"Hanning",    -- 0.013761314529647
      --"Hermite",    -- 0.013850225205266
      "SincFast",   -- 0.014095824314306
      "Jinc",       -- 0.014244299255442
   }
   local downscale_filter = filters[torch.random(1, #filters)]
   local yi = torch.random(INTERPOLATION_PADDING, src:size(2) - size - INTERPOLATION_PADDING)
   local xi = torch.random(INTERPOLATION_PADDING, src:size(3) - size - INTERPOLATION_PADDING)
   local y = src
   local x
   
   if options.color_noise then
      y = color_noise(y)
   end
   if options.overlay then
      y = overlay_augment(y)
   end
   
   x = y
   x = iproc.scale(x, y:size(3) * down_scale, y:size(2) * down_scale, downscale_filter)
   for i = 1, #quality do
      x = gm.Image(x, "RGB", "DHW")
      x:format("jpeg")
      if options.jpeg_sampling_factors == 444 then
	 x:samplingFactors({1.0, 1.0, 1.0})
      else -- 422
	 x:samplingFactors({2.0, 1.0, 1.0})
      end
      local blob, len = x:toBlob(quality[i])
      x:fromBlob(blob, len)
      x = x:toTensor("byte", "RGB", "DHW")
   end
   x = iproc.scale(x, y:size(3), y:size(2))
   y = image.crop(y,
		  xi, yi,
		  xi + size, yi + size)
   x = image.crop(x,
		  xi, yi,
		  xi + size, yi + size)
   x = x:float():div(255)
   y = y:float():div(255)
   x, y = flip_augment(x, y)

   if options.rgb then
   else
      y = image.rgb2yuv(y)[1]:reshape(1, y:size(2), y:size(3))
      x = image.rgb2yuv(x)[1]:reshape(1, x:size(2), x:size(3))
   end
   
   return x, image.crop(y, offset, offset, size - offset, size - offset)
end
function pairwise_transform.jpeg_scale(src, scale, category, level, size, offset, options)
   options = options or {color_noise = false, random_half = true}
   if category == "anime_style_art" then
      if level == 1 then
	 if torch.uniform() > 0.7 then
	    return pairwise_transform.jpeg_scale_(src, scale, {},
						  size, offset, options)
	 else
	    return pairwise_transform.jpeg_scale_(src, scale, {torch.random(65, 85)},
						  size, offset, options)
	 end
      elseif level == 2 then
	 if torch.uniform() > 0.7 then
	    return pairwise_transform.jpeg_scale_(src, scale, {},
						  size, offset, options)
	 else
	    local r = torch.uniform()
	    if r > 0.6 then
	       return pairwise_transform.jpeg_scale_(src, scale, {torch.random(27, 70)},
						     size, offset, options)
	    elseif r > 0.3 then
	       local quality1 = torch.random(37, 70)
	       local quality2 = quality1 - torch.random(5, 10)
	       return pairwise_transform.jpeg_scale_(src, scale, {quality1, quality2},
						     size, offset, options)
	    else
	       local quality1 = torch.random(52, 70)
	       local quality2 = quality1 - torch.random(5, 15)
	       local quality3 = quality1 - torch.random(15, 25)
	       
	       return pairwise_transform.jpeg_scale_(src, scale,
						     {quality1, quality2, quality3 },
						     size, offset, options)
	    end
	 end
      else
	 error("unknown noise level: " .. level)
      end
   elseif category == "photo" then
      if level == 1 then
	 if torch.uniform() > 0.7 then
	    return pairwise_transform.jpeg_scale_(src, scale, {},
						  size, offset, options)
	 else
	 return pairwise_transform.jpeg_scale_(src, scale, {torch.random(80, 95)},
					       size, offset, options)
	 end
      elseif level == 2 then
	 return pairwise_transform.jpeg_scale_(src, scale, {torch.random(70, 85)},
					       size, offset, options)
      else
	 error("unknown noise level: " .. level)
      end
   else
      error("unknown category: " .. category)
   end
end

local function test_jpeg()
   local loader = require './image_loader'
   local src = loader.load_byte("../images/miku_CC_BY-NC.jpg")
   for i = 2, 9 do
      local xy = pairwise_transform.jpeg_(random_half(src),
					  {i * 10}, 128, 0, 2, {color_noise = false, random_half = true, overlay = true, rgb = true})
      for i = 1, #xy do
	 image.display({image = xy[i][1], legend = "y:" .. (i * 10), max=1,min=0})
	 image.display({image = xy[i][2], legend = "x:" .. (i * 10),max=1,min=0})
      end
      --print(x:mean(), y:mean())
   end
end

local function test_scale()
   torch.setdefaulttensortype('torch.FloatTensor')
   local loader = require './image_loader'
   local src = loader.load_byte("../images/miku_CC_BY-NC.jpg")
   local options = {color_noise = true,
		    random_half = true,
		    overlay = false,
		    active_cropping_rate = 1.5,
		    active_cropping_tries = 10,
		    rgb = true
   }
   for i = 1, 9 do
      local xy = pairwise_transform.scale(src, 2.0, 128, 7, 1, options)
      image.display({image = xy[1][1], legend = "y:" .. (i * 10), min = 0, max = 1})
      image.display({image = xy[1][2], legend = "x:" .. (i * 10), min = 0, max = 1})
      print(xy[1][1]:size(), xy[1][2]:size())
      --print(x:mean(), y:mean())
   end
end
local function test_jpeg_scale()
   torch.setdefaulttensortype('torch.FloatTensor')
   local loader = require './image_loader'
   local src = loader.load_byte("../images/miku_CC_BY-NC.jpg")
   local options = {color_noise = true,
		    random_half = true,
		    overlay = true,
		    active_cropping_ratio = 0.5,
		    active_cropping_times = 10
   }
   for i = 1, 9 do
      local y, x = pairwise_transform.jpeg_scale(src, 2.0, 1, 128, 7, options)
      image.display({image = y, legend = "y1:" .. (i * 10), min = 0, max = 1})
      image.display({image = x, legend = "x1:" .. (i * 10), min = 0, max = 1})
      print(y:size(), x:size())
      --print(x:mean(), y:mean())
   end
   for i = 1, 9 do
      local y, x = pairwise_transform.jpeg_scale(src, 2.0, 2, 128, 7, options)
      image.display({image = y, legend = "y2:" .. (i * 10), min = 0, max = 1})
      image.display({image = x, legend = "x2:" .. (i * 10), min = 0, max = 1})
      print(y:size(), x:size())
      --print(x:mean(), y:mean())
   end
end
local function test_color_noise()
   torch.setdefaulttensortype('torch.FloatTensor')
   local loader = require './image_loader'
   local src = loader.load_byte("../images/miku_CC_BY-NC.jpg")
   for i = 1, 10 do
      image.display(color_noise(src))
   end
end
local function test_overlay()
   torch.setdefaulttensortype('torch.FloatTensor')
   local loader = require './image_loader'
   local src = loader.load_byte("../images/miku_CC_BY-NC.jpg")
   for i = 1, 10 do
      image.display(overlay_augment(src, 1.0))
   end
end

--test_scale()
--test_jpeg()
--test_jpeg_scale()
--test_color_noise()
--test_overlay()

return pairwise_transform
