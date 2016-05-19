require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "lib", "?.lua;") .. package.path

require 'image'
local compression = require 'compression'
local settings = require 'settings'
local image_loader = require 'image_loader'
local iproc = require 'iproc'
local alpha_util = require 'alpha_util'

local function crop_if_large(src, max_size)
   local tries = 4
   if src:size(2) >= max_size and src:size(3) >= max_size then
      local rect
      for i = 1, tries do
	 local yi = torch.random(0, src:size(2) - max_size)
	 local xi = torch.random(0, src:size(3) - max_size)
	 rect = iproc.crop(src, xi, yi, xi + max_size, yi + max_size)
	 -- ignore simple background
	 if rect:float():std() >= 0 then
	    break
	 end
      end
      return rect
   else
      return src
   end
end

local function load_images(list)
   local MARGIN = 32
   local lines = utils.split(file.read(list), "\n")
   local x = {}
   local skip_notice = false
   for i = 1, #lines do
      local line = lines[i]
      local im, meta = image_loader.load_byte(line)
      local skip = false
      if meta and meta.alpha then
	 if settings.use_transparent_png then
	    im = alpha_util.fill(im, meta.alpha, torch.random(0, 1))
	 else
	    skip = true
	 end
      end
      if skip then
	 if not skip_notice then
	    io.stderr:write("skip transparent png (settings.use_transparent_png=0)\n")
	    skip_notice = true
	 end
      else
	 if settings.max_training_image_size > 0 then
	    im = crop_if_large(im, settings.max_training_image_size)
	 end
	 im = iproc.crop_mod4(im)
	 local scale = 1.0
	 if settings.random_half_rate > 0.0 then
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
