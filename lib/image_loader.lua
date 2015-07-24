local gm = require 'graphicsmagick'
local ffi = require 'ffi'
require 'pl'

local image_loader = {}

function image_loader.decode_float(blob)
   local im, alpha = image_loader.decode_byte(blob)
   if im then
      im = im:float():div(255)
   end
   return im, alpha
end
function image_loader.encode_png(rgb, alpha)
   if rgb:type() == "torch.ByteTensor" then
      error("expect FloatTensor")
   end
   if alpha then
      if not (alpha:size(2) == rgb:size(2) and  alpha:size(3) == rgb:size(3)) then
	 alpha = gm.Image(alpha, "I", "DHW"):size(rgb:size(3), rgb:size(2), "Sinc"):toTensor("float", "I", "DHW")
      end
      local rgba = torch.Tensor(4, rgb:size(2), rgb:size(3))
      rgba[1]:copy(rgb[1])
      rgba[2]:copy(rgb[2])
      rgba[3]:copy(rgb[3])
      rgba[4]:copy(alpha)
      local im = gm.Image():fromTensor(rgba, "RGBA", "DHW")
      im:format("png")
      return im:toBlob(9)
   else
      local im = gm.Image(rgb, "RGB", "DHW")
      im:format("png")
      return im:toBlob(9)
   end
end
function image_loader.save_png(filename, rgb, alpha)
   local blob, len = image_loader.encode_png(rgb, alpha)
   local fp = io.open(filename, "wb")
   fp:write(ffi.string(blob, len))
   fp:close()
   return true
end
function image_loader.decode_byte(blob)
   local load_image = function()
      local im = gm.Image()
      local alpha = nil
      
      im:fromBlob(blob, #blob)
      -- FIXME: How to detect that a image has an alpha channel?
      if blob:sub(1, 4) == "\x89PNG" or blob:sub(1, 3) == "GIF" then
	 -- split alpha channel
	 im = im:toTensor('float', 'RGBA', 'DHW')
	 local sum_alpha = (im[4] - 1):sum()
	 if sum_alpha > 0 or sum_alpha < 0 then
	    alpha = im[4]:reshape(1, im:size(2), im:size(3))
	 end
	 local new_im = torch.FloatTensor(3, im:size(2), im:size(3))
	 new_im[1]:copy(im[1])
	 new_im[2]:copy(im[2])
	 new_im[3]:copy(im[3])
	 im = new_im:mul(255):byte()
      else
	 im = im:toTensor('byte', 'RGB', 'DHW')
      end
      return {im, alpha}
   end
   local state, ret = pcall(load_image)
   if state then
      return ret[1], ret[2]
   else
      return nil
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
   require 'image'
   local img
   img = image_loader.load_float("./a.jpg")
   if img then
      print(img:min())
      print(img:max())
      image.display(img)
   end
   img = image_loader.load_float("./b.png")
   if img then
      image.display(img)
   end
end
--test()
return image_loader
