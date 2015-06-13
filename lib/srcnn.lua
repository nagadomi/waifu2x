require './LeakyReLU'

function nn.SpatialConvolutionMM:reset(stdv)
   stdv = math.sqrt(2 / ( self.kW * self.kH * self.nOutputPlane))
   self.weight:normal(0, stdv)
   self.bias:fill(0)
end
local srcnn = {}
function srcnn.waifu2x()
   local model = nn.Sequential()
   
   model:add(nn.SpatialConvolutionMM(1, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(32, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(32, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(64, 128, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(128, 128, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(128, 1, 3, 3, 1, 1, 0, 0))
   model:add(nn.View(-1):setNumInputDims(3))
--model:cuda()
--print(model:forward(torch.Tensor(32, 1, 92, 92):uniform():cuda()):size())
   
   return model, 7
end

-- current 4x is worse then 2x * 2
function srcnn.waifu4x()
   local model = nn.Sequential()
   
   model:add(nn.SpatialConvolutionMM(1, 32, 9, 9, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(32, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(32, 64, 5, 5, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(64, 128, 5, 5, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(128, 128, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(128, 1, 5, 5, 1, 1, 0, 0))
   model:add(nn.View(-1):setNumInputDims(3))
   
   return model, 13
end
return srcnn
