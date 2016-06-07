require 'pl'
require 'image'
require 'trepl'

local gm = require 'graphicsmagick'
torch.setdefaulttensortype("torch.FloatTensor")

local function color(black)
   local r, g, b
   if torch.uniform() > 0.8 then
      if black then
	 return {0, 0, 0}
      else
	 return {1, 1, 1}
      end
   else
      if torch.uniform() > 0.7 then
	 r = torch.random(0, 1)
      else
	 r = torch.uniform(0, 1)
      end
      if torch.uniform() > 0.7 then
	 g = torch.random(0, 1)
      else
	 g = torch.uniform(0, 1)
      end
      if torch.uniform() > 0.7 then
	 b = torch.random(0, 1)
      else
	 b = torch.uniform(0, 1)
      end
   end
   return {r,g,b}
end

local function gen_mod()
   local f = function()
      local xm = torch.random(2, 4)
      local ym = torch.random(2, 4)
      return function(x, y) return x % xm == 0 and y % ym == 0 end
   end
   return f()
end
local function dot()
   local sp = 1
   local blocks = {}
   local n = 64
   local s = 24
   for i = 1, n do
      local block = torch.Tensor(3, s, s)
      local margin = torch.random(1, 3)
      local size = torch.random(1, 5)
      local mod = gen_mod()
      local swap_color = torch.uniform() > 0.5
      local fg, bg
      if swap_color then
	 fg = color()
	 bg = color(true)
      else
	 fg = color(true)
	 bg = color()
      end
      local use_cross_and_skip = torch.uniform() > 0.5
      for j = 1, 3 do
	 block[j]:fill(bg[j])
      end
      for y = margin, s - margin do
	 local b = 0
	 if use_cross_and_skip and torch.random(0, 1) == 1 then
	    b = torch.random(0, 1)
	 end
	 for x = margin, s - margin do
	    local yc = math.floor(y / size)
	    local xc = math.floor(x / size)
	    if use_corss_and_skip then
	       if torch.uniform() > 0.25 and mod(yc + b, xc + b) then
		  block[1][y][x] = fg[1]
		  block[2][y][x] = fg[2]
		  block[3][y][x] = fg[3]
	       end
	    else
	       if mod(yc + b, xc + b) then
		  block[1][y][x] = fg[1]
		  block[2][y][x] = fg[2]
		  block[3][y][x] = fg[3]
	       end
	    end
	 end
      end
      block = image.scale(block, s * 2, s * 2, "simple")
      if (not use_corss_and_skip) and size >= 3 and torch.uniform() > 0.5 then
	 block = image.rotate(block, math.pi / 4, "bilinear")
      end
      blocks[i] = block
   end
   local img = torch.Tensor(#blocks, 3, s * 2, s * 2)
   for i = 1, #blocks do
      img[i]:copy(blocks[i])
   end
   img = image.toDisplayTensor({input = img, padding = 0, nrow = math.pow(n, 0.5), min = 0, max = 1})
   return img
end
local function gen()
   return dot()
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text("dot image generator")
cmd:text("Options:")
cmd:option("-o", "", 'output directory')
cmd:option("-n", 64, 'number of images')

local opt = cmd:parse(arg)
if opt.o:len() == 0 then
   cmd:help()
   os.exit(1)
end

for i = 1, opt.n do
   local img = gen()
   image.save(path.join(opt.o, i .. ".png"), img)
end
