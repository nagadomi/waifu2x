require 'image'
require 'cunn'
local iproc = require 'iproc'
local gm = require 'graphicsmagick'
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
      local xcs = torch.LongTensor(tries, y:size(1), size, size)
      local lcs = torch.LongTensor(tries, lowres_y:size(1), size, size)
      local rects = {}
      local r = torch.LongTensor(2, tries)
      r[1]:random(1, x:size(3) - (size + 1)):mul(scale)
      r[2]:random(1, x:size(2) - (size + 1)):mul(scale)
      for i = 1, tries do
	 local xi = r[1][i]
	 local yi = r[2][i]
	 local xc = iproc.crop_nocopy(y, xi, yi, xi + size, yi + size)
	 local lc = iproc.crop_nocopy(lowres_y, xi, yi, xi + size, yi + size)
	 xcs[i]:copy(xc)
	 lcs[i]:copy(lc)
	 rects[i] = {xi, yi}
      end
      xcs:csub(lcs)
      xcs:cmul(xcs)
      local v, l = xcs:reshape(xcs:size(1), xcs:nElement() / xcs:size(1)):transpose(1, 2):sum(1):topk(1, true)
      local best_xi = rects[l[1][1]][1]
      local best_yi = rects[l[1][1]][2]
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
local function lowres_model()
   local seq = nn.Sequential()
   seq:add(nn.SpatialAveragePooling(2, 2, 2, 2))
   seq:add(nn.SpatialUpSamplingNearest(2))
   return seq:cuda()
end
local g_lowres_model = nil
local g_lowres_gpu = nil
function pairwise_transform_utils.low_resolution(src)
   g_lowres_model = g_lowres_model or lowres_model()
   if g_lowres_gpu == nil then
      --benchmark
      local gpu_time = sys.clock()
      for i = 1, 10 do
	 g_lowres_model:forward(src:cuda()):byte()
      end
      gpu_time = sys.clock() - gpu_time

      local cpu_time = sys.clock()
      for i = 1, 10 do
	 gm.Image(src, "RGB", "DHW"):
	    size(src:size(3) * 0.5, src:size(2) * 0.5, "Box"):
	    size(src:size(3), src:size(2), "Box"):
	    toTensor("byte", "RGB", "DHW")
      end
      cpu_time = sys.clock() - cpu_time
      --print(gpu_time, cpu_time)
      if gpu_time < cpu_time then
	 g_lowres_gpu = true
      else
	 g_lowres_gpu = false
      end
   end
   if g_lowres_gpu then
      return g_lowres_model:forward(src:cuda()):byte()
   else
      return gm.Image(src, "RGB", "DHW"):
	 size(src:size(3) * 0.5, src:size(2) * 0.5, "Box"):
	 size(src:size(3), src:size(2), "Box"):
	    toTensor("byte", "RGB", "DHW")
   end
end

return pairwise_transform_utils
