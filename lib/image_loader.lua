local gm = require 'graphicsmagick'
require 'pl'

local image_loader = {}

function image_loader.decode_float(blob)
   local im = image_loader.decode_byte(blob)
   if im then
      im = im:float():div(255)
   end
   return im
end
function image_loader.encode_png(tensor)
   local im = gm.Image(tensor, "RGB", "DHW")
   im:format("png")
   return im:toBlob()
end
function image_loader.decode_byte(blob)
   local load_image = function()
      local im = gm.Image()
      im:fromBlob(blob, #blob)
      -- FIXME: How to detect that a image has an alpha channel?
      if blob:sub(1, 4) == "\x89PNG" or blob:sub(1, 3) == "GIF" then
	 -- merge alpha channel
	 im = im:toTensor('float', 'RGBA', 'DHW')
	 local w2 = im[4]
	 local w1 = im[4] * -1 + 1
	 local new_im = torch.FloatTensor(3, im:size(2), im:size(3))
	 -- apply the white background
	 new_im[1]:copy(im[1]):cmul(w2):add(w1)
	 new_im[2]:copy(im[2]):cmul(w2):add(w1)
	 new_im[3]:copy(im[3]):cmul(w2):add(w1)
	 im = new_im:mul(255):byte()
      else
	 im = im:toTensor('byte', 'RGB', 'DHW')
      end
      return im
   end
   local state, ret = pcall(load_image)
   if state then
      return ret
   else
      return nil
   end
end
function image_loader.load_float(file)
   local fp = io.open(file, "rb")
   local buff = fp:read("*a")
   fp:close()
   return image_loader.decode_float(buff)
end
function image_loader.load_byte(file)
   local fp = io.open(file, "rb")
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
