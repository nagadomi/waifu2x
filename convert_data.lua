require 'torch'
local settings = require './lib/settings'
local image_loader = require './lib/image_loader'

local function count_lines(file)
   local fp = io.open(file, "r")
   local count = 0
   for line in fp:lines() do
      count = count + 1
   end
   fp:close()
   
   return count
end

local function load_images(list)
   local count = count_lines(list)
   local fp = io.open(list, "r")
   local x = {}
   local c = 0
   for line in fp:lines() do
      local im = image_loader.load_byte(line)
      if im then
	 if im:size(2) > settings.crop_size * 2 and im:size(3) > settings.crop_size * 2 then
	    table.insert(x, im)
	 end
      else
	 print("error:" .. line)
      end
      c = c + 1
      xlua.progress(c, count)
      if c % 10 == 0 then
	 collectgarbage()
      end
   end
   return x
end
print(settings)
local x = load_images(settings.image_list)
torch.save(settings.images, x)

