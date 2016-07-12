require 'w2nn'

-- ref: http://arxiv.org/abs/1502.01852
-- ref: http://arxiv.org/abs/1501.00092
local srcnn = {}

function nn.SpatialConvolutionMM:reset(stdv)
   local fin = self.kW * self.kH * self.nInputPlane
   local fout = self.kW * self.kH * self.nOutputPlane
   stdv = math.sqrt(4 / ((1.0 + 0.1 * 0.1) * (fin + fout)))
   self.weight:normal(0, stdv)
   self.bias:zero()
end
function nn.SpatialFullConvolution:reset(stdv)
   local fin = self.kW * self.kH * self.nInputPlane
   local fout = self.kW * self.kH * self.nOutputPlane
   stdv = math.sqrt(4 / ((1.0 + 0.1 * 0.1) * (fin + fout)))
   self.weight:normal(0, stdv)
   self.bias:zero()
end
if cudnn and cudnn.SpatialConvolution then
   function cudnn.SpatialConvolution:reset(stdv)
      local fin = self.kW * self.kH * self.nInputPlane
      local fout = self.kW * self.kH * self.nOutputPlane
      stdv = math.sqrt(4 / ((1.0 + 0.1 * 0.1) * (fin + fout)))
      self.weight:normal(0, stdv)
      self.bias:zero()
   end
   function cudnn.SpatialFullConvolution:reset(stdv)
      local fin = self.kW * self.kH * self.nInputPlane
      local fout = self.kW * self.kH * self.nOutputPlane
      stdv = math.sqrt(4 / ((1.0 + 0.1 * 0.1) * (fin + fout)))
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
   if model.w2nn_channels ~= nil then
      return model.w2nn_channels
   else
      return model:get(model:size() - 1).weight:size(1)
   end
end
function srcnn.backend(model)
   local conv = model:findModules("cudnn.SpatialConvolution")
   local fullconv = model:findModules("cudnn.SpatialFullConvolution")
   if #conv > 0 or #fullconv > 0 then
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
   if model.w2nn_arch_name ~= nil then
      return model.w2nn_arch_name
   else
      local conv = model:findModules("nn.SpatialConvolutionMM")
      if #conv == 0 then
	 conv = model:findModules("cudnn.SpatialConvolution")
      end
      if #conv == 7 then
	 return "vgg_7"
      elseif #conv == 12 then
	 return "vgg_12"
      else
	 error("unsupported model")
      end
   end
end
function srcnn.offset_size(model)
   if model.w2nn_offset ~= nil then
      return model.w2nn_offset
   else
      local name = srcnn.name(model)
      if name:match("vgg_") then
	 local conv = model:findModules("nn.SpatialConvolutionMM")
	 if #conv == 0 then
	    conv = model:findModules("cudnn.SpatialConvolution")
	 end
	 local offset = 0
	 for i = 1, #conv do
	    offset = offset + (conv[i].kW - 1) / 2
	 end
	 return math.floor(offset)
      else
	 error("unsupported model")
      end
   end
end
function srcnn.scale_factor(model)
   if model.w2nn_scale_factor ~= nil then
      return model.w2nn_scale_factor
   else
      local name = srcnn.name(model)
      if name == "upconv_7" then
	 return 2
      elseif name == "upconv_8_4x" then
	 return 4
      else
	 return 1
      end
   end
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
local function SpatialFullConvolution(backend, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, adjW, adjH)
   if backend == "cunn" then
      return nn.SpatialFullConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, adjW, adjH)
   elseif backend == "cudnn" then
      return cudnn.SpatialFullConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   else
      error("unsupported backend:" .. backend)
   end
end

-- VGG style net(7 layers)
function srcnn.vgg_7(backend, ch)
   local model = nn.Sequential()
   model:add(SpatialConvolution(backend, ch, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 32, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 32, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, 128, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 128, 128, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 128, ch, 3, 3, 1, 1, 0, 0))
   model:add(nn.View(-1):setNumInputDims(3))

   model.w2nn_arch_name = "vgg_7"
   model.w2nn_offset = 7
   model.w2nn_scale_factor = 1
   model.w2nn_channels = ch
   --model:cuda()
   --print(model:forward(torch.Tensor(32, ch, 92, 92):uniform():cuda()):size())
   
   return model
end
-- VGG style net(12 layers)
function srcnn.vgg_12(backend, ch)
   local model = nn.Sequential()
   model:add(SpatialConvolution(backend, ch, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 32, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 32, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, 128, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 128, 128, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 128, ch, 3, 3, 1, 1, 0, 0))
   model:add(nn.View(-1):setNumInputDims(3))

   model.w2nn_arch_name = "vgg_12"
   model.w2nn_offset = 12
   model.w2nn_scale_factor = 1
   model.w2nn_resize = false
   model.w2nn_channels = ch
   --model:cuda()
   --print(model:forward(torch.Tensor(32, ch, 92, 92):uniform():cuda()):size())
   
   return model
end

-- Dilated Convolution (7 layers)
function srcnn.dilated_7(backend, ch)
   local model = nn.Sequential()
   model:add(SpatialConvolution(backend, ch, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 32, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(nn.SpatialDilatedConvolution(32, 64, 3, 3, 1, 1, 0, 0, 2, 2))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(nn.SpatialDilatedConvolution(64, 64, 3, 3, 1, 1, 0, 0, 2, 2))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(nn.SpatialDilatedConvolution(64, 128, 3, 3, 1, 1, 0, 0, 4, 4))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 128, 128, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 128, ch, 3, 3, 1, 1, 0, 0))
   model:add(nn.View(-1):setNumInputDims(3))

   model.w2nn_arch_name = "dilated_7"
   model.w2nn_offset = 12
   model.w2nn_scale_factor = 1
   model.w2nn_resize = false
   model.w2nn_channels = ch

   --model:cuda()
   --print(model:forward(torch.Tensor(32, ch, 92, 92):uniform():cuda()):size())
   
   return model
end

-- Upconvolution
function srcnn.upconv_7(backend, ch)
   local model = nn.Sequential()
   model:add(SpatialConvolution(backend, ch, 16, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 16, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 32, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, 128, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 128, 128, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 128, 256, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialFullConvolution(backend, 256, ch, 4, 4, 2, 2, 3, 3):noBias())
   model:add(nn.View(-1):setNumInputDims(3))

   model.w2nn_arch_name = "upconv_7"
   model.w2nn_offset = 14
   model.w2nn_scale_factor = 2
   model.w2nn_resize = true
   model.w2nn_channels = ch

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
   if srcnn[model_name] then
      local model = srcnn[model_name](backend, ch)
      assert(model.w2nn_offset % model.w2nn_scale_factor == 0)
      return model
   else
      error("unsupported model_name: " .. model_name)
   end
end

--local model = srcnn.upconv_6("cunn", 3):cuda()
--print(model:forward(torch.Tensor(1, 3, 64, 64):zero():cuda()):size())

return srcnn
