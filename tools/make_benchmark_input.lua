require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
require 'xlua'
local iproc = require 'iproc'
local image_loader = require 'image_loader'
local gm = require 'graphicsmagick'

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x-make benchmark data")
cmd:text("Options:")

cmd:option("-i", "./data/test", 'input dir')
cmd:option("-lr", "hr", 'highres output dir')
cmd:option("-hr", "lr", 'lowres output dir')
cmd:option("-filter", "Sinc", 'dowsampling filter')

local opt = cmd:parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
local function transform_scale(x, opt)
   return iproc.scale(x,
		      x:size(3) * 0.5,
		      x:size(2) * 0.5,
		      opt.filter, 1)
end
local function load_data_from_dir(test_dir)
   local test_x = {}
   local files = dir.getfiles(test_dir, "*.*")
   for i = 1, #files do
      local name = path.basename(files[i])
      local e = path.extension(name)
      local base = name:sub(0, name:len() - e:len())
      local img = image_loader.load_byte(files[i])
      if img then
	 table.insert(test_x, {y = iproc.crop_mod4(img),
			       basename = base})
      end
      if i % 10 == 0 then
	 if opt.show_progress then
	    xlua.progress(i, #files)
	 end
	 collectgarbage()
      end
   end
   return test_x
end
dir.makepath(opt.lr)
dir.makepath(opt.hr)
local files = load_data_from_dir(opt.i)
for i = 1, #files do
   local y = files[i].y
   local x = transform_scale(y, opt)
   local hr_path = path.join(opt.hr, files[i].basename .. ".png")
   local lr_path = path.join(opt.lr, files[i].basename .. ".png")
   image.save(hr_path, y)
   image.save(lr_path, x)
end
