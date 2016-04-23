require 'w2nn'

-- ref: http://arxiv.org/abs/1502.01852
-- ref: http://arxiv.org/abs/1501.00092
local srcnn = {}

function nn.SpatialConvolutionMM:reset(stdv)
   stdv = math.sqrt(2 / ((1.0 + 0.1 * 0.1) * self.kW * self.kH * self.nOutputPlane))
   self.weight:normal(0, stdv)
   self.bias:zero()
end
if cudnn and cudnn.SpatialConvolution then
   function cudnn.SpatialConvolution:reset(stdv)
      stdv = math.sqrt(2 / ((1.0 + 0.1 * 0.1) * self.kW * self.kH * self.nOutputPlane))
      self.weight:normal(0, stdv)
      self.bias:zero()
   end
end

function nn.SpatialConvolutionMM:clearState()
   if self.gradWeight then
      self.gradWeight:resize(self.nOutputPlane, self.nInputPlane * self.kH * self.kW):zero()
   end
   if self.gradBias then
      self.gradBias:resize(self.nOutputPlane):zero()
   end
   return nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput', 'output', 'gradInput')
end

function srcnn.channels(model)
   return model:get(model:size() - 1).weight:size(1)
end
function srcnn.backend(model)
   local conv = model:findModules("cudnn.SpatialConvolution")
   if #conv > 0 then
      return "cudnn"
   else
      return "cunn"
   end
end
function srcnn.color(model)
   local ch = srcnn.channels(model)
   if ch == 3 then
      return "rgb"
   else
      return "y"
   end
end
function srcnn.name(model)
   local backend_cudnn = false
   local conv = model:findModules("nn.SpatialConvolutionMM")
   if #conv == 0 then
      backend_cudnn = true
      conv = model:findModules("cudnn.SpatialConvolution")
   end
   if #conv == 7 then
      return "vgg_7"
   elseif #conv == 12 then
      return "vgg_12"
   else
      return nil
   end
end
function srcnn.offset_size(model)
   local conv = model:findModules("nn.SpatialConvolutionMM")
   if #conv == 0 then
      conv = model:findModules("cudnn.SpatialConvolution")
   end
   local offset = 0
   for i = 1, #conv do
      offset = offset + (conv[i].kW - 1) / 2
   end
   return math.floor(offset)
end

local function SpatialConvolution(backend, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   if backend == "cunn" then
      return nn.SpatialConvolutionMM(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   elseif backend == "cudnn" then
      return cudnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   else
      error("unsupported backend:" .. backend)
   end
end

-- VGG style net(7 layers)
function srcnn.vgg_7(backend, ch)
   local model = nn.Sequential()
   model:add(SpatialConvolution(backend, ch, 32, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 32, 32, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 32, 64, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 64, 128, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 128, 128, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 128, ch, 3, 3, 1, 1, 0, 0))
   model:add(nn.View(-1):setNumInputDims(3))
   --model:cuda()
   --print(model:forward(torch.Tensor(32, ch, 92, 92):uniform():cuda()):size())
   
   return model
end
-- VGG style net(12 layers)
function srcnn.vgg_12(backend, ch)
   local model = nn.Sequential()
   model:add(SpatialConvolution(backend, ch, 32, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 32, 32, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 32, 64, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 64, 128, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 128, 128, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.LeakyReLU(0.1))
   model:add(SpatialConvolution(backend, 128, ch, 3, 3, 1, 1, 0, 0))
   model:add(nn.View(-1):setNumInputDims(3))
   --model:cuda()
   --print(model:forward(torch.Tensor(32, ch, 92, 92):uniform():cuda()):size())
   
   return model
end

function srcnn.create(model_name, backend, color)
   model_name = model_name or "vgg_7"
   backend = backend or "cunn"
   color = color or "rgb"
   local ch = 3
   if color == "rgb" then
      ch = 3
   elseif color == "y" then
      ch = 1
   else
      error("unsupported color: " .. color)
   end
   if model_name == "vgg_7" then
      return srcnn.vgg_7(backend, ch)
   elseif model_name == "vgg_12" then
      return srcnn.vgg_12(backend, ch)
   else
      error("unsupported model_name: " .. model_name)
   end
end
return srcnn
