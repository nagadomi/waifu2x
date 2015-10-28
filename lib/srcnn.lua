require 'w2nn'

-- ref: http://arxiv.org/abs/1502.01852
-- ref: http://arxiv.org/abs/1501.00092
local srcnn = {}
function srcnn.waifu2x_cunn(ch)
   local model = nn.Sequential()
   model:add(nn.SpatialConvolutionMM(ch, 32, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(32, 32, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(32, 64, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(64, 128, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(128, 128, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(nn.SpatialConvolutionMM(128, ch, 3, 3, 1, 1, 0, 0))
   model:add(nn.View(-1):setNumInputDims(3))
   --model:cuda()
   --print(model:forward(torch.Tensor(32, ch, 92, 92):uniform():cuda()):size())
   
   return model
end
function srcnn.waifu2x_cudnn(ch)
   local model = nn.Sequential()
   model:add(cudnn.SpatialConvolution(ch, 32, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(cudnn.SpatialConvolution(128, ch, 3, 3, 1, 1, 0, 0))
   model:add(nn.View(-1):setNumInputDims(3))
   --model:cuda()
   --print(model:forward(torch.Tensor(32, ch, 92, 92):uniform():cuda()):size())
   
   return model
end
function srcnn.create(model_name, backend, color)
   local ch = 3
   if color == "rgb" then
      ch = 3
   elseif color == "y" then
      ch = 1
   else
      error("unsupported color: " + color)
   end
   if backend == "cunn" then
      return srcnn.waifu2x_cunn(ch)
   elseif backend == "cudnn" then
      return srcnn.waifu2x_cudnn(ch)
   else
      error("unsupported backend: " +  backend)
   end
end
return srcnn
