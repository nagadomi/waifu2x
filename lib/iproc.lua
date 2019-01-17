local gm = {}
gm.Image = require 'graphicsmagick.Image'
require 'dok'
local image = require 'image'
local iproc = {}
local clip_eps8 = (1.0 / 255.0) * 0.5 - (1.0e-7 * (1.0 / 255.0) * 0.5)

function iproc.crop_mod4(src)
   local w = src:size(3) % 4
   local h = src:size(2) % 4
   return iproc.crop(src, 0, 0, src:size(3) - w, src:size(2) - h)
end
function iproc.crop(src, w1, h1, w2, h2)
   local dest
   if src:dim() == 3 then
      dest = src[{{}, { h1 + 1, h2 }, { w1 + 1, w2 }}]:clone()
   else -- dim == 2
      dest = src[{{ h1 + 1, h2 }, { w1 + 1, w2 }}]:clone()
   end
   return dest
end
function iproc.crop_nocopy(src, w1, h1, w2, h2)
   local dest
   if src:dim() == 3 then
      dest = src[{{}, { h1 + 1, h2 }, { w1 + 1, w2 }}]
   else -- dim == 2
      dest = src[{{ h1 + 1, h2 }, { w1 + 1, w2 }}]
   end
   return dest
end
function iproc.byte2float(src)
   local conversion = false
   local dest = src
   if src:type() == "torch.ByteTensor" then
      conversion = true
      dest = src:float():div(255.0)
   end
   return dest, conversion
end
function iproc.float2byte(src)
   local conversion = false
   local dest = src
   if src:type() == "torch.FloatTensor" then
      conversion = true
      dest = (src + clip_eps8):mul(255.0)
      dest:clamp(0, 255.0)
      dest = dest:byte()
   end
   return dest, conversion
end
function iproc.scale(src, width, height, filter, blur)
   local conversion, color
   src, conversion = iproc.byte2float(src)
   filter = filter or "Box"
   if src:size(1) == 3 then
      color = "RGB"
   else
      color = "I"
   end
   local im = gm.Image(src, color, "DHW")
   im:size(math.ceil(width), math.ceil(height), filter, blur)
   local dest = im:toTensor("float", color, "DHW")
   if conversion then
      dest = iproc.float2byte(dest)
   end
   return dest
end
function iproc.scale_with_gamma22(src, width, height, filter, blur)
   local conversion
   src, conversion = iproc.byte2float(src)
   filter = filter or "Box"
   local im = gm.Image(src, "RGB", "DHW")
   im:gammaCorrection(1.0 / 2.2):
      size(math.ceil(width), math.ceil(height), filter, blur):
      gammaCorrection(2.2)
   local dest = im:toTensor("float", "RGB", "DHW"):clamp(0.0, 1.0)
   if conversion then
      dest = iproc.float2byte(dest)
   end
   return dest
end
function iproc.padding(img, w1, w2, h1, h2)
   local conversion
   img, conversion = iproc.byte2float(img)
   image = image or require 'image'
   local dst_height = img:size(2) + h1 + h2
   local dst_width = img:size(3) + w1 + w2
   local flow = torch.Tensor(2, dst_height, dst_width)
   flow[1] = torch.ger(torch.linspace(0, dst_height -1, dst_height), torch.ones(dst_width))
   flow[2] = torch.ger(torch.ones(dst_height), torch.linspace(0, dst_width - 1, dst_width))
   flow[1]:add(-h1)
   flow[2]:add(-w1)
   local dest = image.warp(img, flow, "simple", false, "clamp")
   if conversion then
      dest = iproc.float2byte(dest)
   end
   return dest
end
function iproc.zero_padding(img, w1, w2, h1, h2)
   local conversion
   img, conversion = iproc.byte2float(img)
   image = image or require 'image'
   local dst_height = img:size(2) + h1 + h2
   local dst_width = img:size(3) + w1 + w2
   local flow = torch.Tensor(2, dst_height, dst_width)
   flow[1] = torch.ger(torch.linspace(0, dst_height -1, dst_height), torch.ones(dst_width))
   flow[2] = torch.ger(torch.ones(dst_height), torch.linspace(0, dst_width - 1, dst_width))
   flow[1]:add(-h1)
   flow[2]:add(-w1)
   local dest = image.warp(img, flow, "simple", false, "pad", 0)
   if conversion then
      dest = iproc.float2byte(dest)
   end
   return dest
end
function iproc.white_noise(src, std, rgb_weights, gamma)
   gamma = gamma or 0.454545
   local conversion
   src, conversion = iproc.byte2float(src)
   std = std or 0.01

   local noise = torch.Tensor():resizeAs(src):normal(0, std)
   if rgb_weights then 
      noise[1]:mul(rgb_weights[1])
      noise[2]:mul(rgb_weights[2])
      noise[3]:mul(rgb_weights[3])
   end

   local dest
   if gamma ~= 0 then
      dest = src:clone():pow(gamma):add(noise)
      dest:clamp(0.0, 1.0)
      dest:pow(1.0 / gamma)
   else
      dest = src + noise
   end
   if conversion then
      dest = iproc.float2byte(dest)
   end
   return dest
end
function iproc.hflip(src)
   local t
   if src:type() == "torch.ByteTensor" then
      t = "byte"
   else
      t = "float"
   end
   if src:size(1) == 3 then
      color = "RGB"
   else
      color = "I"
   end
   local im = gm.Image(src, color, "DHW")
   return im:flop():toTensor(t, color, "DHW")
end
function iproc.vflip(src)
   local t
   if src:type() == "torch.ByteTensor" then
      t = "byte"
   else
      t = "float"
   end
   if src:size(1) == 3 then
      color = "RGB"
   else
      color = "I"
   end
   local im = gm.Image(src, color, "DHW")
   return im:flip():toTensor(t, color, "DHW")
end
local function rotate_with_warp(src, dst, theta, mode)
  local height
  local width
  if src:dim() == 2 then
    height = src:size(1)
    width = src:size(2)
  elseif src:dim() == 3 then
    height = src:size(2)
    width = src:size(3)
  else
    dok.error('src image must be 2D or 3D', 'image.rotate')
  end
  local flow = torch.Tensor(2, height, width)
  local kernel = torch.Tensor({{math.cos(-theta), -math.sin(-theta)},
			       {math.sin(-theta), math.cos(-theta)}})
  flow[1] = torch.ger(torch.linspace(0, 1, height), torch.ones(width))
  flow[1]:mul(-(height -1)):add(math.floor(height / 2 + 0.5))
  flow[2] = torch.ger(torch.ones(height), torch.linspace(0, 1, width))
  flow[2]:mul(-(width -1)):add(math.floor(width / 2 + 0.5))
  flow:add(-1, torch.mm(kernel, flow:view(2, height * width)))
  dst:resizeAs(src)
  return image.warp(dst, src, flow, mode, true, 'clamp')
end
function iproc.rotate(src, theta)
   local conversion
   src, conversion = iproc.byte2float(src)
   local dest = torch.Tensor():typeAs(src):resizeAs(src)
   rotate_with_warp(src, dest, theta, 'bilinear')
   dest:clamp(0, 1)
   if conversion then
      dest = iproc.float2byte(dest)
   end
   return dest
end
function iproc.negate(src)
   if src:type() == "torch.ByteTensor" then
      return -src + 255
   else
      return -src + 1
   end
end

function iproc.gaussian2d(kernel_size, sigma)
   sigma = sigma or 1
   local kernel = torch.Tensor(kernel_size, kernel_size)
   local u = math.floor(kernel_size / 2) + 1
   local amp = (1 / math.sqrt(2 * math.pi * sigma^2))
   for x = 1, kernel_size do
      for y = 1, kernel_size do
	 kernel[x][y] = amp * math.exp(-((x - u)^2 + (y - u)^2) / (2 * sigma^2))
      end
   end
   kernel:div(kernel:sum())
   return kernel
end
function iproc.rgb2y(src)
   local conversion
   src, conversion = iproc.byte2float(src)
   local dest = torch.FloatTensor(1, src:size(2), src:size(3)):zero()
   dest:add(0.299, src[1]):add(0.587, src[2]):add(0.114, src[3])
   dest:clamp(0, 1)
   if conversion then
      dest = iproc.float2byte(dest)
   end
   return dest
end

local function test_conversion()
   local a = torch.linspace(0, 255, 256):float():div(255.0)
   local b = iproc.float2byte(a)
   local c = iproc.byte2float(a)
   local d = torch.linspace(0, 255, 256)
   assert((a - c):abs():sum() == 0)
   assert((d:float() - b:float()):abs():sum() == 0)

   a = torch.FloatTensor({256.0, 255.0, 254.999}):div(255.0)
   b = iproc.float2byte(a)
   assert(b:float():sum() == 255.0 * 3)

   a = torch.FloatTensor({254.0, 254.499, 253.50001}):div(255.0)
   b = iproc.float2byte(a)
   print(b)
   assert(b:float():sum() == 254.0 * 3)
end
local function test_flip()
   require 'sys'
   require 'torch'
   torch.setdefaulttensortype("torch.FloatTensor")
   image = require 'image'
   local src = image.lena()
   local src_byte = src:clone():mul(255):byte()

   print(src:size())
   print((image.hflip(src) - iproc.hflip(src)):sum())
   print((image.hflip(src_byte) - iproc.hflip(src_byte)):sum())
   print((image.vflip(src) - iproc.vflip(src)):sum())
   print((image.vflip(src_byte) - iproc.vflip(src_byte)):sum())
end
local function test_gaussian2d()
   local t = {3, 5, 7}
   for i = 1, #t do
      local kp = iproc.gaussian2d(t[i], 0.5)
      print(kp)
   end
end
local function test_conv()
   local image = require 'image'
   local src = image.lena()
   local kernel = torch.Tensor(3, 3):fill(1)
   kernel:div(kernel:sum())
   --local blur = image.convolve(iproc.padding(src, 1, 1, 1, 1), kernel, 'valid')
   local blur = image.convolve(src, kernel, 'same')
   print(src:size(), blur:size())
   local diff = (blur - src):abs()
   image.save("diff.png", diff)
   image.display({image = blur, min=0, max=1})
   image.display({image = diff, min=0, max=1})
end

--test_conversion()
--test_flip()
--test_gaussian2d()
--test_conv()

return iproc


