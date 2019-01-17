-- snapply compression for ByteTensor
local snappy = require 'snappy'

local compression = {}
compression.compress = function (bt)
   local enc = snappy.compress(bt:storage():string())
   return {bt:size(), torch.ByteStorage():string(enc)}
end
compression.decompress = function(data)
   local size = data[1]
   local dec = snappy.decompress(data[2]:string())
   local bt = torch.ByteTensor(table.unpack(torch.totable(size)))
   bt:storage():string(dec)
   return bt
end
compression.size = function(data)
   return data[1]
end

return compression
