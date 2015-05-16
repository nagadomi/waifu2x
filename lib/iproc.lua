local gm = require 'graphicsmagick'
local image = require 'image'
local iproc = {}

function iproc.sample(src, width, height)
   local t = "float"
   if src:type() == "torch.ByteTensor" then
      t = "byte"
   end
   local im = gm.Image(src, "RGB", "DHW")
   im:sample(math.ceil(width), math.ceil(height))
   return im:toTensor(t, "RGB", "DHW")
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
