local gm = require 'graphicsmagick'
local ffi = require 'ffi'
local iproc = require 'iproc'
require 'pl'

local image_loader = {}

local clip_eps8 = (1.0 / 255.0) * 0.5 - (1.0e-7 * (1.0 / 255.0) * 0.5)
local clip_eps16 = (1.0 / 65535.0) * 0.5 - (1.0e-7 * (1.0 / 65535.0) * 0.5)
local background_color = 0.5

function image_loader.encode_png(rgb, depth)
   depth = depth or 8
   rgb = iproc.byte2float(rgb)
   if depth < 16 then
      rgb = rgb:clone():add(clip_eps8)
      rgb[torch.lt(rgb, 0.0)] = 0.0
      rgb[torch.gt(rgb, 1.0)] = 1.0
      rgb = rgb:mul(255):long():float():div(255)
   else
      rgb = rgb:clone():add(clip_eps16)
      rgb[torch.lt(rgb, 0.0)] = 0.0
      rgb[torch.gt(rgb, 1.0)] = 1.0
      rgb = rgb:mul(65535):long():float():div(65535)
   end
   local im
   if rgb:size(1) == 4 then -- RGBA
      im = gm.Image(rgb, "RGBA", "DHW")
   elseif rgb:size(1) == 3 then -- RGB
      im = gm.Image(rgb, "RGB", "DHW")
   elseif rgb:size(1) == 1 then -- Y
      im = gm.Image(rgb, "I", "DHW")
      -- im:colorspace("GRAY") -- it does not work
   end
   return im:depth(depth):format("PNG"):toString(9)
end
function image_loader.save_png(filename, rgb, depth)
   depth = depth or 8
   local blob = image_loader.encode_png(rgb, depth)
   local fp = io.open(filename, "wb")
   if not fp then
      error("IO error: " .. filename)
   end
   fp:write(blob)
   fp:close()
   return true
end
function image_loader.decode_float(blob)
   local load_image = function()
      local im = gm.Image()
      local alpha = nil
      local gamma_lcd = 0.454545
      
      im:fromBlob(blob, #blob)
      
      if im:colorspace() == "CMYK" then
	 im:colorspace("RGB")
      end
      local gamma = math.floor(im:gamma() * 1000000) / 1000000
      if gamma ~= 0 and gamma ~= gamma_lcd then
	 im:gammaCorrection(gamma / gamma_lcd)
      end
      -- FIXME: How to detect that a image has an alpha channel?
      if blob:sub(1, 4) == "\x89PNG" or blob:sub(1, 3) == "GIF" then
	 -- split alpha channel
	 im = im:toTensor('float', 'RGBA', 'DHW')
	 local sum_alpha = (im[4] - 1.0):sum()
	 if sum_alpha < 0 then
	    alpha = im[4]:reshape(1, im:size(2), im:size(3))
	    -- drop full transparent background
	    local mask = torch.le(alpha, 0.0)
	    im[1][mask] = background_color
	    im[2][mask] = background_color
	    im[3][mask] = background_color
	 end
	 local new_im = torch.FloatTensor(3, im:size(2), im:size(3))
	 new_im[1]:copy(im[1])
	 new_im[2]:copy(im[2])
	 new_im[3]:copy(im[3])
	 im = new_im
      else
	 im = im:toTensor('float', 'RGB', 'DHW')
      end
      return {im, alpha, blob}
   end
   local state, ret = pcall(load_image)
   if state then
      return ret[1], ret[2], ret[3]
   else
      return nil, nil, nil
   end
end
function image_loader.decode_byte(blob)
   local im, alpha
   im, alpha, blob = image_loader.decode_float(blob)
   
   if im then
      im = iproc.float2byte(im)
      -- hmm, alpha does not convert here
      return im, alpha, blob
   else
      return nil, nil, nil
   end
end
function image_loader.load_float(file)
   local fp = io.open(file, "rb")
   if not fp then
      error(file .. ": failed to load image")
   end
   local buff = fp:read("*a")
   fp:close()
   return image_loader.decode_float(buff)
end
function image_loader.load_byte(file)
   local fp = io.open(file, "rb")
   if not fp then
      error(file .. ": failed to load image")
   end
   local buff = fp:read("*a")
   fp:close()
   return image_loader.decode_byte(buff)
end
local function test()
   torch.setdefaulttensortype("torch.FloatTensor")
   local a = image_loader.load_float("../images/lena.png")
   local blob = image_loader.encode_png(a)
   local b = image_loader.decode_float(blob)
   assert((b - a):abs():sum() == 0)

   a = image_loader.load_byte("../images/lena.png")
   blob = image_loader.encode_png(a)
   b = image_loader.decode_byte(blob)
   assert((b:float() - a:float()):abs():sum() == 0)
end
--test()
return image_loader
