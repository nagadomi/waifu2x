require './LeakyReLU'

function nn.SpatialConvolutionMM:reset(stdv)
   stdv = math.sqrt(2 / ( self.kW * self.kH * self.nOutputPlane))
   self.weight:normal(0, stdv)
   self.bias:fill(0)
end
local srcnn = {}
function srcnn.waifu2x(color)
   local model = nn.Sequential()
   local ch = nil
   if color == "rgb" then
      ch = 3
   elseif color == "y" then
      ch = 1
   else
      if color then
	 error("unknown color: " .. color)
      else
	 error("unknown color: nil")
      end
   end
   
   model:add(nn.SpatialConvolutionMM(ch, 32, 3, 3, 1, 1, 0, 0))
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
   model:add(nn.SpatialConvolutionMM(128, ch, 3, 3, 1, 1, 0, 0))
   model:add(nn.View(-1):setNumInputDims(3))
--model:cuda()
--print(model:forward(torch.Tensor(32, 1, 92, 92):uniform():cuda()):size())
   
   return model, 7
end

-- current 4x is worse than 2x * 2
function srcnn.waifu4x(color)
   local model = nn.Sequential()

   local ch = nil
   if color == "rgb" then
      ch = 3
   elseif color == "y" then
      ch = 1
   else
      error("unknown color: " .. color)
   end
   
   model:add(nn.SpatialConvolutionMM(ch, 32, 9, 9, 1, 1, 0, 0))
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
   model:add(nn.SpatialConvolutionMM(128, ch, 5, 5, 1, 1, 0, 0))
   model:add(nn.View(-1):setNumInputDims(3))
   
   return model, 13
end
return srcnn
