local gm = require 'graphicsmagick'
local image = require 'image'

local iproc = {}

function iproc.crop_mod4(src)
   local w = src:size(3) % 4
   local h = src:size(2) % 4
   return image.crop(src, 0, 0, src:size(3) - w, src:size(2) - h)
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
      dest = (src * 255.0):byte()
   end
   return dest, conversion
end
function iproc.scale(src, width, height, filter)
   local t = "float"
   if src:type() == "torch.ByteTensor" then
      t = "byte"
   end
   filter = filter or "Box"
   local im = gm.Image(src, "RGB", "DHW")
   im:size(math.ceil(width), math.ceil(height), filter)
   return im:toTensor(t, "RGB", "DHW")
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


return iproc
