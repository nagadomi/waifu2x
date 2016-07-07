require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "lib", "?.lua;") .. package.path

require 'image'
local cjson = require 'cjson'
local csvigo = require 'csvigo'
local compression = require 'compression'
local settings = require 'settings'
local image_loader = require 'image_loader'
local iproc = require 'iproc'
local alpha_util = require 'alpha_util'

local function crop_if_large(src, max_size)
   if max_size < 0 then
      return src
   end
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
local function crop_if_large_pair(x, y, max_size)
   if max_size < 0 then
      return x, y
   end
   local scale_y = y:size(2) / x:size(2)
   local mod = 4
   assert(x:size(3) == (y:size(3) / scale_y))

   local tries = 4
   if y:size(2) > max_size and y:size(3) > max_size then
      assert(max_size % 4 == 0)
      local rect_x, rect_y
      for i = 1, tries do
	 local yi = torch.random(0, y:size(2) - max_size)
	 local xi = torch.random(0, y:size(3) - max_size)
	 if mod then
	    yi = yi - (yi % mod)
	    xi = xi - (xi % mod)
	 end
	 rect_y = iproc.crop(y, xi, yi, xi + max_size, yi + max_size)
	 rect_x = iproc.crop(y, xi / scale_y, yi / scale_y, xi / scale_y + max_size / scale_y, yi / scale_y + max_size / scale_y)
	 -- ignore simple background
	 if rect_y:float():std() >= 0 then
	    break
	 end
      end
      return rect_x, rect_y
   else
      return x, y
   end
end

local function load_images(list)
   local MARGIN = 32
   local csv = csvigo.load({path = list, verbose = false, mode = "raw"})
   local x = {}
   local skip_notice = false
   for i = 1, #csv do
      local filename = csv[i][1]
      local csv_meta = csv[i][2]
      if csv_meta and csv_meta:len() > 0 then
	 csv_meta = cjson.decode(csv_meta)
      end
      if csv_meta and csv_meta.filters then
	 filters = csv_meta.filters
      end
      local im, meta = image_loader.load_byte(filename)
      local skip = false
      local alpha_color = torch.random(0, 1)

      if meta and meta.alpha then
	 if settings.use_transparent_png then
	    im = alpha_util.fill(im, meta.alpha, alpha_color)
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
	 if csv_meta and csv_meta.x then
	    -- method == user
	    local yy = im
	    local xx, meta2 = image_loader.load_byte(csv_meta.x)
	    if meta2 and meta2.alpha then
	       xx = alpha_util.fill(xx, meta2.alpha, alpha_color)
	    end
	    xx, yy = crop_if_large_pair(xx, yy, settings.max_training_image_size)
	    table.insert(x, {{y = compression.compress(yy), x = compression.compress(xx)},
			    {data = {filters = filters, has_x = true}}})
	 else
	    im = crop_if_large(im, settings.max_training_image_size)
	    im = iproc.crop_mod4(im)
	    local scale = 1.0
	    if settings.random_half_rate > 0.0 then
	       scale = 2.0
	    end
	    if im then
	       if im:size(2) > (settings.crop_size * scale + MARGIN) and im:size(3) > (settings.crop_size * scale + MARGIN) then
		  table.insert(x, {compression.compress(im), {data = {filters = filters}}})
	       else
		  io.stderr:write(string.format("\n%s: skip: image is too small (%d > size).\n", filename, settings.crop_size * scale + MARGIN))
	       end
	    else
	       io.stderr:write(string.format("\n%s: skip: load error.\n", filename))
	    end
	 end
      end
      xlua.progress(i, #csv)
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
