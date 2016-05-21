local gm = require 'graphicsmagick'
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
      dest[torch.lt(dest, 0.0)] = 0
      dest[torch.gt(dest, 255.0)] = 255.0
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
   local dst_height = img:size(2) + h1 + h2
   local dst_width = img:size(3) + w1 + w2
   local flow = torch.Tensor(2, dst_height, dst_width)
   flow[1] = torch.ger(torch.linspace(0, dst_height -1, dst_height), torch.ones(dst_width))
   flow[2] = torch.ger(torch.ones(dst_height), torch.linspace(0, dst_width - 1, dst_width))
   flow[1]:add(-h1)
   flow[2]:add(-w1)
   return image.warp(img, flow, "simple", false, "clamp")
end
function iproc.zero_padding(img, w1, w2, h1, h2)
   local dst_height = img:size(2) + h1 + h2
   local dst_width = img:size(3) + w1 + w2
   local flow = torch.Tensor(2, dst_height, dst_width)
   flow[1] = torch.ger(torch.linspace(0, dst_height -1, dst_height), torch.ones(dst_width))
   flow[2] = torch.ger(torch.ones(dst_height), torch.linspace(0, dst_width - 1, dst_width))
   flow[1]:add(-h1)
   flow[2]:add(-w1)
   return image.warp(img, flow, "simple", false, "pad", 0)
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
      dest[torch.lt(dest, 0.0)] = 0.0
      dest[torch.gt(dest, 1.0)] = 1.0
      dest:pow(1.0 / gamma)
   else
      dest = src + noise
   end
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
--test_conversion()

return iproc
