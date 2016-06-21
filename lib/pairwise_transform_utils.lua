require 'image'
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
function pairwise_transform_utils.crop_if_large(src, max_size, mod)
   local tries = 4
   if src:size(2) > max_size and src:size(3) > max_size then
      assert(max_size % 4 == 0)
      local rect
      for i = 1, tries do
	 local yi = torch.random(0, src:size(2) - max_size)
	 local xi = torch.random(0, src:size(3) - max_size)
	 if mod then
	    yi = yi - (yi % mod)
	    xi = xi - (xi % mod)
	 end
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
   local box_only = false
   if options.data.filters then
      if #options.data.filters == 1 and options.data.filters[1] == "Box" then
	 box_only = true
      end
   end
   if box_only then
      local mod = 2 -- assert pos % 2 == 0
      dest = pairwise_transform_utils.crop_if_large(dest, math.max(crop_size * 2, options.max_size), mod)
      dest = data_augmentation.flip(dest)
      dest = data_augmentation.color_noise(dest, options.random_color_noise_rate)
      dest = data_augmentation.overlay(dest, options.random_overlay_rate)
      dest = data_augmentation.unsharp_mask(dest, options.random_unsharp_mask_rate)
      dest = iproc.crop_mod4(dest)
   else
      dest = pairwise_transform_utils.random_half(dest, options.random_half_rate, options.downsampling_filters)
      dest = pairwise_transform_utils.crop_if_large(dest, math.max(crop_size * 2, options.max_size))
      dest = data_augmentation.flip(dest)
      dest = data_augmentation.color_noise(dest, options.random_color_noise_rate)
      dest = data_augmentation.overlay(dest, options.random_overlay_rate)
      dest = data_augmentation.unsharp_mask(dest, options.random_unsharp_mask_rate)
      dest = data_augmentation.shift_1px(dest)
   end
   return dest
end
function pairwise_transform_utils.active_cropping(x, y, lowres_y, size, scale, p, tries)
   assert("x:size == y:size", x:size(2) * scale == y:size(2) and x:size(3) * scale == y:size(3))
   assert("crop_size % scale == 0", size % scale == 0)
   local r = torch.uniform()
   local t = "float"
   if x:type() == "torch.ByteTensor" then
      t = "byte"
   end
   if p < r then
      local xi = torch.random(1, x:size(3) - (size + 1)) * scale
      local yi = torch.random(1, x:size(2) - (size + 1)) * scale
      local yc = iproc.crop(y, xi, yi, xi + size, yi + size)
      local xc = iproc.crop(x, xi / scale, yi / scale, xi / scale + size / scale, yi / scale + size / scale)
      return xc, yc
   else
      local best_se = 0.0
      local best_xi, best_yi
      local m = torch.LongTensor(y:size(1), size, size)
      local targets = {}
      for i = 1, tries do
	 local xi = torch.random(1, x:size(3) - (size + 1)) * scale
	 local yi = torch.random(1, x:size(2) - (size + 1)) * scale
	 local xc = iproc.crop_nocopy(y, xi, yi, xi + size, yi + size)
	 local lc = iproc.crop_nocopy(lowres_y, xi, yi, xi + size, yi + size)
	 m:copy(xc:long()):csub(lc:long())
	 m:cmul(m)
	 local se = m:sum()
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
function pairwise_transform_utils.flip_augmentation(x, y, lowres_y, x_noise)
   local xs = {}
   local ns = {}
   local ys = {}
   local ls = {}

   for j = 1, 2 do
      -- TTA
      local xi, yi, ri
      if j == 1 then
	 xi = x
	 ni = x_noise
	 yi = y
	 ri = lowres_y
      else
	 xi = x:transpose(2, 3):contiguous()
	 if x_noise then
	    ni = x_noise:transpose(2, 3):contiguous()
	 end
	 yi = y:transpose(2, 3):contiguous()
	 ri = lowres_y:transpose(2, 3):contiguous()
      end
      local xv = image.vflip(xi)
      local nv
      if x_noise then
	 nv = image.vflip(ni)
      end
      local yv = image.vflip(yi)
      local rv = image.vflip(ri)
      table.insert(xs, xi)
      if ni then
	 table.insert(ns, ni)
      end
      table.insert(ys, yi)
      table.insert(ls, ri)

      table.insert(xs, xv)
      if nv then
	 table.insert(ns, nv)
      end
      table.insert(ys, yv)
      table.insert(ls, rv)

      table.insert(xs, image.hflip(xi))
      if ni then
	 table.insert(ns, image.hflip(ni))
      end
      table.insert(ys, image.hflip(yi))
      table.insert(ls, image.hflip(ri))

      table.insert(xs, image.hflip(xv))
      if nv then
	 table.insert(ns, image.hflip(nv))
      end
      table.insert(ys, image.hflip(yv))
      table.insert(ls, image.hflip(rv))
   end
   return xs, ys, ls, ns
end

return pairwise_transform_utils
