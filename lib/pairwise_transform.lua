require 'image'
local gm = require 'graphicsmagick'
local iproc = require './iproc'
local reconstruct = require './reconstruct'
local pairwise_transform = {}

local function random_half(src, p, min_size)
   p = p or 0.5
   local filter = ({"Box","Blackman", "SincFast", "Jinc"})[torch.random(1, 4)]
   if p > torch.uniform() then
      return iproc.scale(src, src:size(3) * 0.5, src:size(2) * 0.5, filter)
   else
      return src
   end
end
local function color_augment(x)
   local color_scale = torch.Tensor(3):uniform(0.8, 1.2)
   x = x:float():div(255)
   for i = 1, 3 do
      x[i]:mul(color_scale[i])
   end
   x[torch.lt(x, 0.0)] = 0.0
   x[torch.gt(x, 1.0)] = 1.0
   return x:mul(255):byte()
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
local INTERPOLATION_PADDING = 16
function pairwise_transform.scale(src, scale, size, offset, options)
   options = options or {color_augment = true, random_half = true, rgb = true}
   if options.random_half then
      src = random_half(src)
   end
   local yi = torch.random(INTERPOLATION_PADDING, src:size(2) - size - INTERPOLATION_PADDING)
   local xi = torch.random(INTERPOLATION_PADDING, src:size(3) - size - INTERPOLATION_PADDING)
   local down_scale = 1.0 / scale
   local y = image.crop(src,
			xi - INTERPOLATION_PADDING, yi - INTERPOLATION_PADDING,
			xi + size + INTERPOLATION_PADDING, yi + size + INTERPOLATION_PADDING)
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
   
   y = flip_augment(y)
   if options.color_augment then
      y = color_augment(y)
   end
   local x = iproc.scale(y, y:size(3) * down_scale, y:size(2) * down_scale, downscale_filter)
   x = iproc.scale(x, y:size(3), y:size(2))
   y = y:float():div(255)
   x = x:float():div(255)

   if options.rgb then
   else
      y = image.rgb2yuv(y)[1]:reshape(1, y:size(2), y:size(3))
      x = image.rgb2yuv(x)[1]:reshape(1, x:size(2), x:size(3))
   end

   y = image.crop(y, INTERPOLATION_PADDING + offset, INTERPOLATION_PADDING + offset, y:size(3) - offset -	INTERPOLATION_PADDING, y:size(2) - offset - INTERPOLATION_PADDING)
   x = image.crop(x, INTERPOLATION_PADDING, INTERPOLATION_PADDING, x:size(3) - INTERPOLATION_PADDING, x:size(2) - INTERPOLATION_PADDING)
   
   return x, y
end
function pairwise_transform.jpeg_(src, quality, size, offset, options)
   options = options or {color_augment = true, random_half = true, rgb = true}
   if options.random_half then
      src = random_half(src)
   end
   local yi = torch.random(0, src:size(2) - size - 1)
   local xi = torch.random(0, src:size(3) - size - 1)
   local y = src
   local x

   if options.color_augment then
      y = color_augment(y)
   end
   x = y
   for i = 1, #quality do
      x = gm.Image(x, "RGB", "DHW")
      x:format("jpeg")
      x:samplingFactors({1.0, 1.0, 1.0})
      local blob, len = x:toBlob(quality[i])
      x:fromBlob(blob, len)
      x = x:toTensor("byte", "RGB", "DHW")
   end
   
   y = image.crop(y, xi, yi, xi + size, yi + size)
   x = image.crop(x, xi, yi, xi + size, yi + size)
   y = y:float():div(255)
   x = x:float():div(255)
   x, y = flip_augment(x, y)
   
   if options.rgb then
   else
      y = image.rgb2yuv(y)[1]:reshape(1, y:size(2), y:size(3))
      x = image.rgb2yuv(x)[1]:reshape(1, x:size(2), x:size(3))
   end
   
   return x, image.crop(y, offset, offset, size - offset, size - offset)
end
function pairwise_transform.jpeg(src, level, size, offset, options)
   if level == 1 then
      return pairwise_transform.jpeg_(src, {torch.random(65, 85)},
				      size, offset,
				      options)
   elseif level == 2 then
      local r = torch.uniform()
      if r > 0.6 then
	 return pairwise_transform.jpeg_(src, {torch.random(27, 70)},
					 size, offset,
					 options)
      elseif r > 0.3 then
	 local quality1 = torch.random(37, 70)
	 local quality2 = quality1 - torch.random(5, 10)
	 return pairwise_transform.jpeg_(src, {quality1, quality2},
					    size, offset,
					    options)
      else
	 local quality1 = torch.random(52, 70)
	 return pairwise_transform.jpeg_(src,
					 {quality1,
					  quality1 - torch.random(5, 15),
					  quality1 - torch.random(15, 25)},
					 size, offset,
					 options)
      end
   else
      error("unknown noise level: " .. level)
   end
end
function pairwise_transform.jpeg_scale_(src, scale, quality, size, offset, options)
   if options.random_half then
      src = random_half(src)
   end
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
   
   if options.color_augment then
      y = color_augment(y)
   end
   x = y
   x = iproc.scale(x, y:size(3) * down_scale, y:size(2) * down_scale, downscale_filter)
   for i = 1, #quality do
      x = gm.Image(x, "RGB", "DHW")
      x:format("jpeg")
      x:samplingFactors({1.0, 1.0, 1.0})
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
function pairwise_transform.jpeg_scale(src, scale, level, size, offset, options)
   options = options or {color_augment = true, random_half = true}
   if level == 1 then
      return pairwise_transform.jpeg_scale_(src, scale, {torch.random(65, 85)},
					    size, offset, options)
   elseif level == 2 then
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
	    return pairwise_transform.jpeg_scale_(src, scale,
						  {quality1,
						   quality1 - torch.random(5, 15),
						   quality1 - torch.random(15, 25)},
						  size, offset, options)
      end
   else
      error("unknown noise level: " .. level)
   end
end

local function test_jpeg()
   local loader = require './image_loader'
   local src = loader.load_byte("../images/miku_CC_BY-NC.jpg")
   local y, x = pairwise_transform.jpeg_(src, {}, 128, 0, false)
   image.display({image = y, legend = "y:0"})
   image.display({image = x, legend = "x:0"})
   for i = 2, 9 do
      local y, x = pairwise_transform.jpeg_(pairwise_transform.random_half(src),
					    {i * 10}, 128, 0, {color_augment = false, random_half = true})
      image.display({image = y, legend = "y:" .. (i * 10), max=1,min=0})
      image.display({image = x, legend = "x:" .. (i * 10),max=1,min=0})
      --print(x:mean(), y:mean())
   end
end

local function test_scale()
   local loader = require './image_loader'
   local src = loader.load_byte("../images/miku_CC_BY-NC.jpg")   
   for i = 1, 9 do
      local y, x = pairwise_transform.scale(src, 2.0, 128, 7, {color_augment = true, random_half = true, rgb = true})
      image.display({image = y, legend = "y:" .. (i * 10), min = 0, max = 1})
      image.display({image = x, legend = "x:" .. (i * 10), min = 0, max = 1})
      print(y:size(), x:size())
      --print(x:mean(), y:mean())
   end
end
local function test_jpeg_scale()
   local loader = require './image_loader'
   local src = loader.load_byte("../images/miku_CC_BY-NC.jpg")   
   for i = 1, 9 do
      local y, x = pairwise_transform.jpeg_scale(src, 2.0, 1, 128, 7, {color_augment = true, random_half = true})
      image.display({image = y, legend = "y1:" .. (i * 10), min = 0, max = 1})
      image.display({image = x, legend = "x1:" .. (i * 10), min = 0, max = 1})
      print(y:size(), x:size())
      --print(x:mean(), y:mean())
   end
   for i = 1, 9 do
      local y, x = pairwise_transform.jpeg_scale(src, 2.0, 2, 128, 7, {color_augment = true, random_half = true})
      image.display({image = y, legend = "y2:" .. (i * 10), min = 0, max = 1})
      image.display({image = x, legend = "x2:" .. (i * 10), min = 0, max = 1})
      print(y:size(), x:size())
      --print(x:mean(), y:mean())
   end
end
--test_scale()
--test_jpeg()
--test_jpeg_scale()

return pairwise_transform
