require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path
require 'os'
require 'w2nn'
local srcnn = require 'srcnn'

local function find_aux(seq)
   for k = 1, #seq.modules do
      local mod = seq.modules[k]
      local name = torch.typename(mod)
      if name == "nn.Sequential" or name == "nn.ConcatTable" then
	 local aux = find_aux(mod)
	 if aux ~= nil then
	    return aux
	 end
      elseif name == "w2nn.AuxiliaryLossTable" then
	 return mod
      end
   end
   return nil
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text("switch the output pass of auxiliary loss")
cmd:text("Options:")
cmd:option("-j", 1, 'Specify the output path index (1|2)')
cmd:option("-i", "", 'Specify the input model')
cmd:option("-o", "", 'Specify the output model')
cmd:option("-iformat", "ascii", 'Specify the input format (ascii|binary)')
cmd:option("-oformat", "ascii", 'Specify the output format (ascii|binary)')

local opt = cmd:parse(arg)
if not path.isfile(opt.i) then
   cmd:help()
   os.exit(-1)
end

local model = torch.load(opt.i, opt.iformat)
if model == nil then
   print("load error")
   os.exit(-1)
end
local aux = find_aux(model)
if aux == nil then
   print("AuxiliaryLossTable not found")
else
   print(aux)
   aux.i = opt.j
   torch.save(opt.o, model, opt.oformat)
end
