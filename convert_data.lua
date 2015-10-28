local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "lib", "?.lua;") .. package.path

require 'pl'
require 'image'
local compression = require 'compression'
local settings = require 'settings'
local image_loader = require 'image_loader'

local MAX_SIZE = 1440

local function crop_if_large(src, max_size)
   if max_size > 0 and (src:size(2) > max_size or src:size(3) > max_size) then
      local sx = torch.random(0, src:size(3) - math.min(max_size, src:size(3)))
      local sy = torch.random(0, src:size(2) - math.min(max_size, src:size(2)))
      return image.crop(src, sx, sy,
			math.min(sx + max_size, src:size(3)),
			math.min(sy + max_size, src:size(2)))
   else
      return src
   end
end
local function crop_4x(x)
   local w = x:size(3) % 4
   local h = x:size(2) % 4
   return image.crop(x, 0, 0, x:size(3) - w, x:size(2) - h)
end

local function load_images(list)
   local MARGIN = 32
   local lines = utils.split(file.read(list), "\n")
   local x = {}
   for i = 1, #lines do
      local line = lines[i]
      local im, alpha = image_loader.load_byte(line)
      if alpha then
	 io.stderr:write(string.format("\n%s: skip: image has alpha channel.\n", line))
      else
	 im = crop_if_large(im, settings.max_size)
	 im = crop_4x(im)
	 local scale = 1.0
	 if settings.random_half then
	    scale = 2.0
	 end
	 if im then
	    if im:size(2) > (settings.crop_size * scale + MARGIN) and im:size(3) > (settings.crop_size * scale + MARGIN) then
	       table.insert(x, compression.compress(im))
	    else
	       io.stderr:write(string.format("\n%s: skip: image is too small (%d > size).\n", line, settings.crop_size * scale + MARGIN))
	    end
	 else
	    io.stderr:write(string.format("\n%s: skip: load error.\n", line))
	 end
      end
      xlua.progress(i, #lines)
      if i % 10 == 0 then
	 collectgarbage()
      end
   end
   return x
end

torch.manualSeed(settings.seed)
print(settings)
local x = load_images(settings.image_list)
torch.save(settings.images, x)
