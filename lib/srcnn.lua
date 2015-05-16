require 'cunn'
require 'cudnn'
require './LeakyReLU'

function cudnn.SpatialConvolution:reset(stdv)
   stdv = math.sqrt(2 / ( self.kW * self.kH * self.nOutputPlane))
   self.weight:normal(0, stdv)
   self.bias:fill(0)
end
local function create_model()
   local model = nn.Sequential() 
   
   model:add(cudnn.SpatialConvolution(1, 32, 3, 3, 1, 1, 0, 0):fastest())
   model:add(nn.LeakyReLU(0.1))   
   model:add(cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1, 0, 0):fastest())
   model:add(nn.LeakyReLU(0.1))
   model:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, 0, 0):fastest())
   model:add(nn.LeakyReLU(0.1))
   model:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 0, 0):fastest())
   model:add(nn.LeakyReLU(0.1))
   model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 0, 0):fastest())
   model:add(nn.LeakyReLU(0.1))
   model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 0, 0):fastest())
   model:add(nn.LeakyReLU(0.1))
   model:add(cudnn.SpatialConvolution(128, 1, 3, 3, 1, 1, 0, 0):fastest())
   model:add(nn.View(-1):setNumInputDims(3))
--model:cuda()
--print(model:forward(torch.Tensor(32, 1, 92, 92):uniform():cuda()):size())
   
   return model, 7
end
return create_model
