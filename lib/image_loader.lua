local gm = require 'graphicsmagick'
local ffi = require 'ffi'
local iproc = require 'iproc'
local sRGB2014 = require 'sRGB2014'
require 'pl'

local image_loader = {}
local clip_eps8 = (1.0 / 255.0) * 0.5 - (1.0e-7 * (1.0 / 255.0) * 0.5)
local clip_eps16 = (1.0 / 65535.0) * 0.5 - (1.0e-7 * (1.0 / 65535.0) * 0.5)
local background_color = 0.5

function image_loader.encode_png(rgb, options)
   options = options or {}
   options.depth = options.depth or 8
   if options.inplace == nil then
      options.inplace = false
   end
   rgb = iproc.byte2float(rgb)
   if options.depth < 16 then
      if options.inplace then
	 rgb:add(clip_eps8)
      else
	 rgb = rgb:clone():add(clip_eps8)
      end
      rgb:clamp(0.0, 1.0)
      rgb = rgb:mul(255):floor():div(255)
   else
      if options.inplace then
	 rgb:add(clip_eps16)
      else
	 rgb = rgb:clone():add(clip_eps16)
      end
      rgb:clamp(0.0, 1.0)
      rgb = rgb:mul(65535):floor():div(65535)
   end
   local im
   if rgb:size(1) == 4 then -- RGBA
      im = gm.Image(rgb, "RGBA", "DHW")
      if options.grayscale then
	 im:type("GrayscaleMatte")
      end
   elseif rgb:size(1) == 3 then -- RGB
      im = gm.Image(rgb, "RGB", "DHW")
      if options.grayscale then
	 im:type("Grayscale")
      end
   elseif rgb:size(1) == 1 then -- Y
      im = gm.Image(rgb, "I", "DHW")
      im:type("Grayscale")
   end
   if options.gamma then
      im:gamma(options.gamma)
   end
   if options.icm and im.profile then
      im:profile("icm", sRGB2014)
      im:profile("icm", options.icm)
   end
   return im:depth(options.depth):format("PNG"):toString()
end
function image_loader.save_png(filename, rgb, options)
   local blob = image_loader.encode_png(rgb, options)
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
      local meta = {}
      local im = gm.Image()
      local gamma_lcd = 0.454545
      
      im:fromBlob(blob, #blob)
      if im.profile then
	 meta.icm = im:profile("icm")
	 if meta.icm then
	    im:profile("icm", sRGB2014)
	    im:removeProfile()
	 end
      end
      if im:colorspace() == "CMYK" then
	 im:colorspace("RGB")
      end
      if gamma ~= 0 and math.floor(im:gamma() * 1000000) / 1000000 ~= gamma_lcd then
	 meta.gamma = im:gamma()
      end
      local image_type = im:type()
      if image_type == "Grayscale" or image_type == "GrayscaleMatte" then
	 meta.grayscale = true
      end
      if image_type == "TrueColorMatte" or image_type == "GrayscaleMatte" then
	 -- split alpha channel
	 im = im:toTensor('float', 'RGBA', 'DHW')
	 meta.alpha = im[4]:reshape(1, im:size(2), im:size(3))
	 -- drop full transparent background
	 local mask = torch.le(meta.alpha, 0.0)
	 im[1][mask] = background_color
	 im[2][mask] = background_color
	 im[3][mask] = background_color
	 local new_im = torch.FloatTensor(3, im:size(2), im:size(3))
	 new_im[1]:copy(im[1])
	 new_im[2]:copy(im[2])
	 new_im[3]:copy(im[3])
	 im = new_im
      else
	 im = im:toTensor('float', 'RGB', 'DHW')
      end
      meta.blob = blob
      return {im, meta}
   end
   local state, ret = pcall(load_image)
   if state then
      return ret[1], ret[2]
   else
      return nil, nil
   end
end
function image_loader.decode_byte(blob)
   local im, meta
   im, meta = image_loader.decode_float(blob)
   
   if im then
      im = iproc.float2byte(im)
      -- hmm, alpha does not convert here
      return im, meta
   else
      return nil, nil
   end
end
function image_loader.load_float(file)
   local fp = io.open(file, "rb")
   if not fp then
      error(file .. ": failed to load image")
   end
   local buff = fp:read("*all")
   fp:close()
   return image_loader.decode_float(buff)
end
function image_loader.load_byte(file)
   local fp = io.open(file, "rb")
   if not fp then
      error(file .. ": failed to load image")
   end
   local buff = fp:read("*all")
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
