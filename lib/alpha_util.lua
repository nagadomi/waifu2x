local w2nn = require 'w2nn'
local reconstruct = require 'reconstruct'
local image = require 'image'
local iproc = require 'iproc'
local gm = require 'graphicsmagick'

alpha_util = {}

function alpha_util.make_border(rgb, alpha, offset)
   if not alpha then
      return rgb
   end
   local sum2d = nn.SpatialConvolutionMM(1, 1, 3, 3, 1, 1, 1, 1):cuda()
   sum2d.weight:fill(1)
   sum2d.bias:zero()

   local mask = alpha:clone()
   mask[torch.gt(mask, 0.0)] = 1
   mask[torch.eq(mask, 0.0)] = 0
   local mask_nega = (mask - 1):abs():byte()
   local eps = 1.0e-7

   rgb = rgb:clone()
   rgb[1][mask_nega] = 0
   rgb[2][mask_nega] = 0
   rgb[3][mask_nega] = 0

   for i = 1, offset do
      local mask_weight = sum2d:forward(mask:cuda()):float()
      local border = rgb:clone()
      for j = 1, 3 do
	 border[j]:copy(sum2d:forward(rgb[j]:reshape(1, rgb:size(2), rgb:size(3)):cuda()))
	 border[j]:cdiv((mask_weight + eps))
	 rgb[j][mask_nega] = border[j][mask_nega]
      end
      mask = mask_weight:clone()
      mask[torch.gt(mask_weight, 0.0)] = 1
      mask_nega = (mask - 1):abs():byte()
      if border:size(2) * border:size(3) > 1024*1024 then
	 collectgarbage()
      end
   end
   rgb:clamp(0.0, 1.0)

   return rgb
end
function alpha_util.composite(rgb, alpha, model2x)
   if not alpha then
      return rgb
   end
   if not (alpha:size(2) == rgb:size(2) and  alpha:size(3) == rgb:size(3)) then
      if model2x then
	 alpha = reconstruct.scale(model2x, 2.0, alpha)
      else
	 alpha = gm.Image(alpha, "I", "DHW"):size(rgb:size(3), rgb:size(2), "Sinc"):toTensor("float", "I", "DHW")
      end
   end
   local out = torch.Tensor(4, rgb:size(2), rgb:size(3))
   out[1]:copy(rgb[1])
   out[2]:copy(rgb[2])
   out[3]:copy(rgb[3])
   out[4]:copy(alpha)
   return out
end
function alpha_util.fill(fg, alpha, val)
   assert(fg:size(2) == alpha:size(2) and fg:size(3) == alpha:size(3))
   local conversion = false
   fg, conversion = iproc.byte2float(fg)
   val = val or 0
   fg = fg:clone()
   bg = fg:clone():fill(val)
   bg[1]:cmul(1-alpha)
   bg[2]:cmul(1-alpha)
   bg[3]:cmul(1-alpha)
   fg[1]:cmul(alpha)
   fg[2]:cmul(alpha)
   fg[3]:cmul(alpha)

   local ret = bg:add(fg)
   if conversion then
      ret = iproc.float2byte(ret)
   end
   return ret
end

local function test()
   require 'sys'
   require 'trepl'
   torch.setdefaulttensortype("torch.FloatTensor")

   local image_loader = require 'image_loader'
   local rgb, alpha = image_loader.load_float("alpha.png")
   local t = sys.clock()
   rgb = alpha_util.make_border(rgb, alpha, 7)
   print(sys.clock() - t)
   print(rgb:min(), rgb:max())
   image.display({image = rgb, min = 0, max = 1})
   image.save("out.png", rgb)
end
--test()

return alpha_util
