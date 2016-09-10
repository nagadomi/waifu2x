local gm = {}
gm.Image = require 'graphicsmagick.Image'
local image = nil

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
   image = image or require 'image'
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
   image = image or require 'image'
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

-- from torch/image
----------------------------------------------------------------------
-- image.rgb2yuv(image)
-- converts a RGB image to YUV
--
function iproc.rgb2yuv(...)
   -- arg check
   local output,input
   local args = {...}
   if select('#',...) == 2 then
      output = args[1]
      input = args[2]
   elseif select('#',...) == 1 then
      input = args[1]
   else
      print(dok.usage('image.rgb2yuv',
                      'transforms an image from RGB to YUV', nil,
                      {type='torch.Tensor', help='input image', req=true},
                      '',
                      {type='torch.Tensor', help='output image', req=true},
                      {type='torch.Tensor', help='input image', req=true}
                      ))
      dok.error('missing input', 'image.rgb2yuv')
   end

   -- resize
   output = output or input.new()
   output:resizeAs(input)

   -- input chanels
   local inputRed = input[1]
   local inputGreen = input[2]
   local inputBlue = input[3]

   -- output chanels
   local outputY = output[1]
   local outputU = output[2]
   local outputV = output[3]

   -- convert
   outputY:zero():add(0.299, inputRed):add(0.587, inputGreen):add(0.114, inputBlue)
   outputU:zero():add(-0.14713, inputRed):add(-0.28886, inputGreen):add(0.436, inputBlue)
   outputV:zero():add(0.615, inputRed):add(-0.51499, inputGreen):add(-0.10001, inputBlue)

   -- return YUV image
   return output
end

----------------------------------------------------------------------
-- image.yuv2rgb(image)
-- converts a YUV image to RGB
--
function iproc.yuv2rgb(...)
   -- arg check
   local output,input
   local args = {...}
   if select('#',...) == 2 then
      output = args[1]
      input = args[2]
   elseif select('#',...) == 1 then
      input = args[1]
   else
      print(dok.usage('image.yuv2rgb',
                      'transforms an image from YUV to RGB', nil,
                      {type='torch.Tensor', help='input image', req=true},
                      '',
                      {type='torch.Tensor', help='output image', req=true},
                      {type='torch.Tensor', help='input image', req=true}
                      ))
      dok.error('missing input', 'image.yuv2rgb')
   end

   -- resize
   output = output or input.new()
   output:resizeAs(input)

   -- input chanels
   local inputY = input[1]
   local inputU = input[2]
   local inputV = input[3]

   -- output chanels
   local outputRed = output[1]
   local outputGreen = output[2]
   local outputBlue = output[3]

   -- convert
   outputRed:copy(inputY):add(1.13983, inputV)
   outputGreen:copy(inputY):add(-0.39465, inputU):add(-0.58060, inputV)
   outputBlue:copy(inputY):add(2.03211, inputU)

   -- return RGB image
   return output
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

--test_conversion()
--test_flip()

return iproc


