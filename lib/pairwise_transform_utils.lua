require 'image'
local gm = require 'graphicsmagick'
local iproc = require 'iproc'
local data_augmentation = require 'data_augmentation'
local pairwise_transform_utils = {}

function pairwise_transform_utils.random_half(src, p, filters)
   if torch.uniform() < p then
      local filter = filters[torch.random(1, #filters)]
      return iproc.scale(src, src:size(3) * 0.5, src:size(2) * 0.5, filter)
   else
      return src
   end
end
function pairwise_transform_utils.crop_if_large(src, max_size)
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
function pairwise_transform_utils.preprocess(src, crop_size, options)
   local dest = src
   dest = pairwise_transform_utils.random_half(dest, options.random_half_rate, options.downsampling_filters)
   dest = pairwise_transform_utils.crop_if_large(dest, math.max(crop_size * 2, options.max_size))
   dest = data_augmentation.flip(dest)
   dest = data_augmentation.color_noise(dest, options.random_color_noise_rate)
   dest = data_augmentation.overlay(dest, options.random_overlay_rate)
   dest = data_augmentation.unsharp_mask(dest, options.random_unsharp_mask_rate)
   dest = data_augmentation.shift_1px(dest)
   
   return dest
end
function pairwise_transform_utils.active_cropping(x, y, size, scale, p, tries)
   assert("x:size == y:size", x:size(2) * scale == y:size(2) and x:size(3) * scale == y:size(3))
   assert("crop_size % scale == 0", size % scale == 0)
   local r = torch.uniform()
   local t = "float"
   if x:type() == "torch.ByteTensor" then
      t = "byte"
   end
   if p < r then
      local xi = torch.random(0, x:size(3) - (size + 1))
      local yi = torch.random(0, x:size(2) - (size + 1))
      local yc = iproc.crop(y, xi * scale, yi * scale, xi * scale + size, yi * scale + size)
      local xc = iproc.crop(x, xi, yi, xi + size / scale, yi + size / scale)
      return xc, yc
   else
      local lowres = gm.Image(y, "RGB", "DHW"):
	    size(y:size(3) * 0.5, y:size(2) * 0.5, "Box"):
	    size(y:size(3), y:size(2), "Box"):
	    toTensor(t, "RGB", "DHW")
      local best_se = 0.0
      local best_xi, best_yi
      local m = torch.FloatTensor(y:size(1), size, size)
      for i = 1, tries do
	 local xi = torch.random(0, x:size(3) - (size + 1)) * scale
	 local yi = torch.random(0, x:size(2) - (size + 1)) * scale
	 local xc = iproc.crop(y, xi, yi, xi + size, yi + size)
	 local lc = iproc.crop(lowres, xi, yi, xi + size, yi + size)
	 local xcf = iproc.byte2float(xc)
	 local lcf = iproc.byte2float(lc)
	 local se = m:copy(xcf):add(-1.0, lcf):pow(2):sum()
	 if se >= best_se then
	    best_xi = xi
	    best_yi = yi
	    best_se = se
	 end
      end
      local yc = iproc.crop(y, best_xi, best_yi, best_xi + size, best_yi + size)
      local xc = iproc.crop(x, best_xi / scale, best_yi / scale, best_xi / scale + size / scale, best_yi / scale + size / scale)
      return xc, yc
   end
end

return pairwise_transform_utils
