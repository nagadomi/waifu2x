require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "..", "lib", "?.lua;") .. package.path

require 'w2nn'
torch.setdefaulttensortype("torch.FloatTensor")

local cmd = torch.CmdLine()
cmd:text()
cmd:text("cleanup model")
cmd:text("Options:")
cmd:option("-model", "./model.t7", 'path of model file')
cmd:option("-iformat", "binary", 'input format')
cmd:option("-oformat", "binary", 'output format')

local opt = cmd:parse(arg)
local model = torch.load(opt.model, opt.iformat)
if model then
   w2nn.cleanup_model(model)
   torch.save(opt.model, model, opt.oformat)
else
   error("model not found")
end
