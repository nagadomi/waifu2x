require 'image'
local gm = require 'graphicsmagick'
local iproc = require './iproc'
local reconstract = require './reconstract'
local pairwise_transform = {}

function pairwise_transform.scale(src, scale, size, offset, options)
   options = options or {}
   local yi = torch.radom(0, src:size(2) - size - 1)
   local xi = torch.random(0, src:size(3) - size - 1)
   local down_scale = 1.0 / scale
   local y = image.crop(src, xi, yi, xi + size, yi + size)
   local flip = torch.random(1, 4)
   local nega = torch.random(0, 1)
   local filters = {
      "Box",        -- 0.012756949974688
      "Blackman",   -- 0.013191924552285
      --"Cartom",     -- 0.013753536746706
      --"Hanning",    -- 0.013761314529647
      --"Hermite",    -- 0.013850225205266
      --"SincFast",   -- 0.014095824314306
      --"Jinc",       -- 0.014244299255442
   }
   local downscale_filter = filters[torch.random(1, #filters)]
   
   if r == 1 then
      y = image.hflip(y)
   elseif r == 2 then
      y = image.vflip(y)
   elseif r == 3 then
      y = image.hflip(image.vflip(y))
   elseif r == 4 then
      -- none
   end
   if options.color_augment then
      y = y:float():div(255)
      local color_scale = torch.Tensor(3):uniform(0.8, 1.2)
      for i = 1, 3 do
	 y[i]:mul(color_scale[i])
      end
      y[torch.lt(y, 0)] = 0
      y[torch.gt(y, 1.0)] = 1.0
      y = y:mul(255):byte()
   end
   local x = iproc.scale(y, y:size(3) * down_scale, y:size(2) * down_scale, downscale_filter)
   if options.noise and (options.noise_ratio or 0.5) > torch.uniform() then
      -- add noise
      local quality = {torch.random(70, 90)}
      for i = 1, #quality do
	 x = gm.Image(x, "RGB", "DHW")
	 x:format("jpeg")
	 local blob, len = x:toBlob(quality[i])
	 x:fromBlob(blob, len)
	    x = x:toTensor("byte", "RGB", "DHW")
      end
   end
   if options.denoise_model and (options.denoise_ratio or 0.5) > torch.uniform() then
      x = reconstract(options.denoise_model, x:float():div(255), offset):mul(255):byte()
   end
   x = iproc.scale(x, y:size(3), y:size(2))
   y = y:float():div(255)
   x = x:float():div(255)
   y = image.rgb2yuv(y)[1]:reshape(1, y:size(2), y:size(3))
   x = image.rgb2yuv(x)[1]:reshape(1, x:size(2), x:size(3))
   
   return x, image.crop(y, offset, offset, size - offset, size - offset)
end
function pairwise_transform.jpeg_(src, quality, size, offset, color_augment)
   if color_augment == nil then color_augment = true end
   local yi = torch.random(0, src:size(2) - size - 1)
   local xi = torch.random(0, src:size(3) - size - 1)
   local y = src
   local x
   local flip = torch.random(1, 4)

   if color_augment then
      local color_scale = torch.Tensor(3):uniform(0.8, 1.2)
      y = y:float():div(255)
      for i = 1, 3 do
	 y[i]:mul(color_scale[i])
      end
      y[torch.lt(y, 0)] = 0
      y[torch.gt(y, 1.0)] = 1.0
      y = y:mul(255):byte()
   end
   x = y
   for i = 1, #quality do
      x = gm.Image(x, "RGB", "DHW")
      x:format("jpeg")
      local blob, len = x:toBlob(quality[i])
      x:fromBlob(blob, len)
      x = x:toTensor("byte", "RGB", "DHW")
   end
   
   y = image.crop(y, xi, yi, xi + size, yi + size)
   x = image.crop(x, xi, yi, xi + size, yi + size)
   x = x:float():div(255)
   y = y:float():div(255)
   
   if flip == 1 then
      y = image.hflip(y)
      x = image.hflip(x)
   elseif flip == 2 then
      y = image.vflip(y)
      x = image.vflip(x)
   elseif flip == 3 then
      y = image.hflip(image.vflip(y))
      x = image.hflip(image.vflip(x))
   elseif flip == 4 then
      -- none
   end
   y = image.rgb2yuv(y)[1]:reshape(1, y:size(2), y:size(3))
   x = image.rgb2yuv(x)[1]:reshape(1, x:size(2), x:size(3))

   return x, image.crop(y, offset, offset, size - offset, size - offset)
end
function pairwise_transform.jpeg(src, level, size, offset, color_augment)
   if level == 1 then
      return pairwise_transform.jpeg_(src, {torch.random(65, 85)},
				      size, offset,
				      color_augment)
   elseif level == 2 then
      local r = torch.uniform()
      if r > 0.6 then
	 return pairwise_transform.jpeg_(src, {torch.random(27, 80)},
					 size, offset,
					 color_augment)
      elseif r > 0.3 then
	 local quality1 = torch.random(32, 40)
	 local quality2 = quality1 - 5
	 return pairwise_transform.jpeg_(src, {quality1, quality2},
					 size, offset,
					 color_augment)
      else
	 local quality1 = torch.random(47, 70)
	 return pairwise_transform.jpeg_(src, {quality1, quality1 - 10, quality1 - 20},
					 size, offset,
					 color_augment)
      end
   else
      error("unknown noise level: " .. level)
   end
end

local function test_jpeg()
   local loader = require 'image_loader'
   local src = loader.load_byte("a.jpg")

   for i = 2, 9 do
      local y, x = pairwise_transform.jpeg_(src, {i * 10}, 128, 0, false)
      image.display({image = y, legend = "y:" .. (i * 10), max=1,min=0})
      image.display({image = x, legend = "x:" .. (i * 10),max=1,min=0})
      --print(x:mean(), y:mean())
   end
end
local function test_scale()
   require 'nn'
   require 'cudnn'
   require './LeakyReLU'
   
   local loader = require 'image_loader'
   local src = loader.load_byte("e.jpg")

   for i = 1, 9 do
      local y, x = pairwise_transform.scale(src, 2.0, "Box", 128, 7, {noise = true, denoise_model = torch.load("models/noise1_model.t7")})
      image.display({image = y, legend = "y:" .. (i * 10)})
      image.display({image = x, legend = "x:" .. (i * 10)})
      --print(x:mean(), y:mean())
   end
end
--test_jpeg()
--test_scale()

return pairwise_transform
