require 'pl'
require 'cunn'
local iproc = require 'iproc'
local gm = {}
gm.Image = require 'graphicsmagick.Image'
local data_augmentation = {}

local function pcacov(x)
   local mean = torch.mean(x, 1)
   local xm = x - torch.ger(torch.ones(x:size(1)), mean:squeeze())
   local c = torch.mm(xm:t(), xm)
   c:div(x:size(1) - 1)
   local ce, cv = torch.symeig(c, 'V')
   return ce, cv
end

function random_rect_size(rect_min, rect_max)
   local r = torch.Tensor(2):uniform():cmul(torch.Tensor({rect_max - rect_min, rect_max - rect_min})):int()
   local rect_h = r[1] + rect_min
   local rect_w = r[2] + rect_min
   return rect_h, rect_w
end
function random_rect(height, width, rect_h, rect_w)
   local r = torch.Tensor(2):uniform():cmul(torch.Tensor({height - 1 - rect_h, width-1 - rect_w})):int()
   local rect_y1 = r[1] + 1
   local rect_x1 = r[2] + 1
   local rect_x2 = rect_x1 + rect_w
   local rect_y2 = rect_y1 + rect_h
   return {x1 = rect_x1, y1 = rect_y1, x2 = rect_x2, y2 = rect_y2}
end
function data_augmentation.erase(src, p, n, rect_min, rect_max)
   if torch.uniform() < p then
      local src, conversion = iproc.byte2float(src)
      src = src:contiguous():clone()
      local ch = src:size(1)
      local height = src:size(2)
      local width = src:size(3)
      for i = 1, n do
	 local rect_h, rect_w = random_rect_size(rect_min, rect_max)
	 local rect1 = random_rect(height, width, rect_h, rect_w)
	 local rect2 = random_rect(height, width, rect_h, rect_w)
	 dest_rect = src:sub(1, ch, rect1.y1, rect1.y2, rect1.x1, rect1.x2)
	 src_rect = src:sub(1, ch, rect2.y1, rect2.y2, rect2.x1, rect2.x2)
	 dest_rect:copy(src_rect:clone())
      end
      if conversion then
	 src = iproc.float2byte(src)
      end
      return src
   else
      return src
   end
end
function data_augmentation.color_noise(src, p, factor)
   factor = factor or 0.1
   if torch.uniform() < p then
      local src, conversion = iproc.byte2float(src)
      local src_t = src:reshape(src:size(1), src:nElement() / src:size(1)):t():contiguous()
      local ce, cv = pcacov(src_t)
      local color_scale = torch.Tensor(3):uniform(1 / (1 + factor), 1 + factor)
      
      pca_space = torch.mm(src_t, cv):t():contiguous()
      for i = 1, 3 do
	 pca_space[i]:mul(color_scale[i])
      end
      local dest = torch.mm(pca_space:t(), cv:t()):t():contiguous():resizeAs(src)
      dest:clamp(0.0, 1.0)

      if conversion then
	 dest = iproc.float2byte(dest)
      end
      return dest
   else
      return src
   end
end
function data_augmentation.overlay(src, p)
   if torch.uniform() < p then
      local r = torch.uniform()
      local src, conversion = iproc.byte2float(src)
      src = src:contiguous()
      local flip = data_augmentation.flip(src)
      flip:mul(r):add(src * (1.0 - r))
      if conversion then
	 flip = iproc.float2byte(flip)
      end
      return flip
   else
      return src
   end
end
function data_augmentation.unsharp_mask(src, p)
   if torch.uniform() < p then
      local radius = 0 -- auto
      local sigma = torch.uniform(0.5, 1.5)
      local amount = torch.uniform(0.1, 0.9)
      local threshold = torch.uniform(0.0, 0.05)
      local unsharp = gm.Image(src, "RGB", "DHW"):
	 unsharpMask(radius, sigma, amount, threshold):
	 toTensor("float", "RGB", "DHW")
      
      if src:type() == "torch.ByteTensor" then
	 return iproc.float2byte(unsharp)
      else
	 return unsharp
      end
   else
      return src
   end
end
function data_augmentation.blur(src, p, size, sigma_min, sigma_max)
   size = size or "3"
   filters = utils.split(size, ",")
   for i = 1, #filters do
      local s = tonumber(filters[i])
      filters[i] = s
   end
   if torch.uniform() < p then
      local src, conversion = iproc.byte2float(src)
      local kernel_size = filters[torch.random(1, #filters)]
      local sigma
      if sigma_min == sigma_max then
	 sigma = sigma_min
      else
	 sigma = torch.uniform(sigma_min, sigma_max)
      end
      local kernel = iproc.gaussian2d(kernel_size, sigma)
      local dest = image.convolve(src, kernel, 'same')
      if conversion then
	 dest = iproc.float2byte(dest)
      end
      return dest
   else
      return src
   end
end
function data_augmentation.pairwise_scale(x, y, p, scale_min, scale_max)
   if torch.uniform() < p then
      assert(x:size(2) == y:size(2) and x:size(3) == y:size(3))
      local scale = torch.uniform(scale_min, scale_max)
      local h = math.floor(x:size(2) * scale)
      local w = math.floor(x:size(3) * scale)
      local filters = {"Lanczos", "Catrom"}
      local x_filter = filters[torch.random(1, 2)]
      x = iproc.scale(x, w, h, x_filter)
      y = iproc.scale(y, w, h, "Triangle")
      return x, y
   else
      return x, y
   end
end
function data_augmentation.pairwise_rotate(x, y, p, r_min, r_max)
   if torch.uniform() < p then
      assert(x:size(2) == y:size(2) and x:size(3) == y:size(3))
      local r = torch.uniform(r_min, r_max) / 360.0 * math.pi
      x = iproc.rotate(x, r)
      y = iproc.rotate(y, r)
      return x, y
   else
      return x, y
   end
end
function data_augmentation.pairwise_negate(x, y, p)
   if torch.uniform() < p then
      assert(x:size(2) == y:size(2) and x:size(3) == y:size(3))
      x = iproc.negate(x)
      y = iproc.negate(y)
      return x, y
   else
      return x, y
   end
end
function data_augmentation.pairwise_negate_x(x, y, p)
   if torch.uniform() < p then
      assert(x:size(2) == y:size(2) and x:size(3) == y:size(3))
      x = iproc.negate(x)
      return x, y
   else
      return x, y
   end
end
function data_augmentation.pairwise_flip(x, y)
   local flip = torch.random(1, 4)
   local tr = torch.random(1, 2)
   local x, conversion = iproc.byte2float(x)
   y = iproc.byte2float(y)
   x = x:contiguous()
   y = y:contiguous()
   if tr == 1 then
      -- pass
   elseif tr == 2 then
      x = x:transpose(2, 3):contiguous()
      y = y:transpose(2, 3):contiguous()
   end
   if flip == 1 then
      x = iproc.hflip(x)
      y = iproc.hflip(y)
   elseif flip == 2 then
      x = iproc.vflip(x)
      y = iproc.vflip(y)
   elseif flip == 3 then
      x = iproc.hflip(iproc.vflip(x))
      y = iproc.hflip(iproc.vflip(y))
   elseif flip == 4 then
   end
   if conversion then
      x = iproc.float2byte(x)
      y = iproc.float2byte(y)
   end
   return x, y
end
function data_augmentation.shift_1px(src)
   -- reducing the even/odd issue in nearest neighbor scaler.
   local direction = torch.random(1, 4)
   local x_shift = 0
   local y_shift = 0
   if direction == 1 then
      x_shift = 1
      y_shift = 0
   elseif direction == 2 then
      x_shift = 0
      y_shift = 1
   elseif direction == 3 then
      x_shift = 1
      y_shift = 1
   elseif flip == 4 then
      x_shift = 0
      y_shift = 0
   end
   local w = src:size(3) - x_shift
   local h = src:size(2) - y_shift
   w = w - (w % 4)
   h = h - (h % 4)
   local dest = iproc.crop(src, x_shift, y_shift, x_shift + w, y_shift + h)
   return dest
end
function data_augmentation.flip(src)
   local flip = torch.random(1, 4)
   local tr = torch.random(1, 2)
   local src, conversion = iproc.byte2float(src)
   local dest
   src = src:contiguous()
   if tr == 1 then
      -- pass
   elseif tr == 2 then
      src = src:transpose(2, 3):contiguous()
   end
   if flip == 1 then
      dest = iproc.hflip(src)
   elseif flip == 2 then
      dest = iproc.vflip(src)
   elseif flip == 3 then
      dest = iproc.hflip(iproc.vflip(src))
   elseif flip == 4 then
      dest = src
   end
   if conversion then
      dest = iproc.float2byte(dest)
   end
   return dest
end

local function test_blur()
   torch.setdefaulttensortype("torch.FloatTensor")
   local image =require 'image'
   local src = image.lena()

   image.display({image = src, min=0, max=1})
   local dest = data_augmentation.blur(src, 1.0, "3,5", 0.5, 0.6)
   image.display({image = dest, min=0, max=1})
   dest = data_augmentation.blur(src, 1.0, "3", 1.0, 1.0)
   image.display({image = dest, min=0, max=1})
   dest = data_augmentation.blur(src, 1.0, "5", 0.75, 0.75)
   image.display({image = dest, min=0, max=1})
end
--test_blur()

return data_augmentation
