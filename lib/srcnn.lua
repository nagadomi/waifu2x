require 'w2nn'

-- ref: http://arxiv.org/abs/1502.01852
-- ref: http://arxiv.org/abs/1501.00092
local srcnn = {}

local function msra_filler(mod)
   local fin = mod.kW * mod.kH * mod.nInputPlane
   local fout = mod.kW * mod.kH * mod.nOutputPlane
   stdv = math.sqrt(4 / ((1.0 + 0.1 * 0.1) * (fin + fout)))
   mod.weight:normal(0, stdv)
   mod.bias:zero()
end
local function identity_filler(mod)
   assert(mod.nInputPlane <= mod.nOutputPlane)
   mod.weight:normal(0, 0.01)
   mod.bias:zero()
   local num_groups = mod.nInputPlane -- fixed
   local filler_value = num_groups / mod.nOutputPlane
   local in_group_size = math.floor(mod.nInputPlane / num_groups)
   local out_group_size = math.floor(mod.nOutputPlane / num_groups)
   local x = math.floor(mod.kW / 2)
   local y = math.floor(mod.kH / 2)
   for i = 0, num_groups - 1 do
      for j = i * out_group_size, (i + 1) * out_group_size - 1 do
	 for k = i * in_group_size, (i + 1) * in_group_size - 1 do
	    mod.weight[j+1][k+1][y+1][x+1] = filler_value
	 end
      end
   end
end
function nn.SpatialConvolutionMM:reset(stdv)
   msra_filler(self)
end
function nn.SpatialFullConvolution:reset(stdv)
   msra_filler(self)
end
function nn.SpatialDilatedConvolution:reset(stdv)
   identity_filler(self)
end

if cudnn and cudnn.SpatialConvolution then
   function cudnn.SpatialConvolution:reset(stdv)
      msra_filler(self)
   end
   function cudnn.SpatialFullConvolution:reset(stdv)
      msra_filler(self)
   end
   if cudnn.SpatialDilatedConvolution then
      function cudnn.SpatialDilatedConvolution:reset(stdv)
	 identity_filler(self)
      end
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
srcnn.SpatialConvolution = SpatialConvolution

local function SpatialFullConvolution(backend, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, adjW, adjH)
   if backend == "cunn" then
      return nn.SpatialFullConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, adjW, adjH)
   elseif backend == "cudnn" then
      return cudnn.SpatialFullConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   else
      error("unsupported backend:" .. backend)
   end
end
srcnn.SpatialFullConvolution = SpatialFullConvolution

local function ReLU(backend)
   if backend == "cunn" then
      return nn.ReLU(true)
   elseif backend == "cudnn" then
      return cudnn.ReLU(true)
   else
      error("unsupported backend:" .. backend)
   end
end
srcnn.ReLU = ReLU

local function SpatialMaxPooling(backend, kW, kH, dW, dH, padW, padH)
   if backend == "cunn" then
      return nn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
   elseif backend == "cudnn" then
      return cudnn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
   else
      error("unsupported backend:" .. backend)
   end
end
srcnn.SpatialMaxPooling = SpatialMaxPooling

local function SpatialAveragePooling(backend, kW, kH, dW, dH, padW, padH)
   if backend == "cunn" then
      return nn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH)
   elseif backend == "cudnn" then
      return cudnn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH)
   else
      error("unsupported backend:" .. backend)
   end
end
srcnn.SpatialAveragePooling = SpatialAveragePooling

local function SpatialDilatedConvolution(backend, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, dilationW, dilationH)      
   if backend == "cunn" then
      return nn.SpatialDilatedConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, dilationW, dilationH)
   elseif backend == "cudnn" then
      if cudnn.SpatialDilatedConvolution then
	 -- cudnn v 6
	 return cudnn.SpatialDilatedConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, dilationW, dilationH)
      else
	 return nn.SpatialDilatedConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, dilationW, dilationH)
      end
   else
      error("unsupported backend:" .. backend)
   end
end
srcnn.SpatialDilatedConvolution = SpatialDilatedConvolution


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
   model:add(w2nn.InplaceClip01())
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
   model:add(w2nn.InplaceClip01())
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
   model:add(w2nn.InplaceClip01())
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
   model:add(w2nn.InplaceClip01())
   model:add(nn.View(-1):setNumInputDims(3))


   model.w2nn_arch_name = "upconv_7"
   model.w2nn_offset = 14
   model.w2nn_scale_factor = 2
   model.w2nn_resize = true
   model.w2nn_channels = ch

   return model
end

-- large version of upconv_7
-- This model able to beat upconv_7 (PSNR: +0.3 ~ +0.8) but this model is 2x slower than upconv_7.
function srcnn.upconv_7l(backend, ch)
   local model = nn.Sequential()
   model:add(SpatialConvolution(backend, ch, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 32, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, 128, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 128, 192, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 192, 256, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 256, 512, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialFullConvolution(backend, 512, ch, 4, 4, 2, 2, 3, 3):noBias())
   model:add(w2nn.InplaceClip01())
   model:add(nn.View(-1):setNumInputDims(3))

   model.w2nn_arch_name = "upconv_7l"
   model.w2nn_offset = 14
   model.w2nn_scale_factor = 2
   model.w2nn_resize = true
   model.w2nn_channels = ch

   --model:cuda()
   --print(model:forward(torch.Tensor(32, ch, 92, 92):uniform():cuda()):size())

   return model
end

-- layerwise linear blending with skip connections
-- Note: PSNR: upconv_7 < skiplb_7 < upconv_7l
function srcnn.skiplb_7(backend, ch)
   local function skip(backend, i, o)
      local con = nn.Concat(2)
      local conv = nn.Sequential()
      conv:add(SpatialConvolution(backend, i, o, 3, 3, 1, 1, 1, 1))
      conv:add(nn.LeakyReLU(0.1, true))

      -- depth concat
      con:add(conv)
      con:add(nn.Identity()) -- skip
      return con
   end
   local model = nn.Sequential()
   model:add(skip(backend, ch, 16))
   model:add(skip(backend, 16+ch, 32))
   model:add(skip(backend, 32+16+ch, 64))
   model:add(skip(backend, 64+32+16+ch, 128))
   model:add(skip(backend, 128+64+32+16+ch, 128))
   model:add(skip(backend, 128+128+64+32+16+ch, 256))
   -- input of last layer = [all layerwise output(contains input layer)].flatten
   model:add(SpatialFullConvolution(backend, 256+128+128+64+32+16+ch, ch, 4, 4, 2, 2, 3, 3):noBias()) -- linear blend
   model:add(w2nn.InplaceClip01())
   model:add(nn.View(-1):setNumInputDims(3))
   model.w2nn_arch_name = "skiplb_7"
   model.w2nn_offset = 14
   model.w2nn_scale_factor = 2
   model.w2nn_resize = true
   model.w2nn_channels = ch

   --model:cuda()
   --print(model:forward(torch.Tensor(32, ch, 92, 92):uniform():cuda()):size())

   return model
end

-- dilated convolution + deconvolution
-- Note: This model is not better than upconv_7. Maybe becuase of under-fitting.
function srcnn.dilated_upconv_7(backend, ch)
   local model = nn.Sequential()
   model:add(SpatialConvolution(backend, ch, 16, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 16, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(nn.SpatialDilatedConvolution(32, 64, 3, 3, 1, 1, 0, 0, 2, 2))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(nn.SpatialDilatedConvolution(64, 128, 3, 3, 1, 1, 0, 0, 2, 2))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(nn.SpatialDilatedConvolution(128, 128, 3, 3, 1, 1, 0, 0, 2, 2))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 128, 256, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialFullConvolution(backend, 256, ch, 4, 4, 2, 2, 3, 3):noBias())
   model:add(w2nn.InplaceClip01())
   model:add(nn.View(-1):setNumInputDims(3))

   model.w2nn_arch_name = "dilated_upconv_7"
   model.w2nn_offset = 20
   model.w2nn_scale_factor = 2
   model.w2nn_resize = true
   model.w2nn_channels = ch

   --model:cuda()
   --print(model:forward(torch.Tensor(32, ch, 92, 92):uniform():cuda()):size())

   return model
end

-- ref: https://arxiv.org/abs/1609.04802
-- note: no batch-norm, no zero-paading
function srcnn.srresnet_2x(backend, ch)
   local function resblock(backend)
      local seq = nn.Sequential()
      local con = nn.ConcatTable()
      local conv = nn.Sequential()
      conv:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
      conv:add(ReLU(backend))
      conv:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
      conv:add(ReLU(backend))
      con:add(conv)
      con:add(nn.SpatialZeroPadding(-2, -2, -2, -2)) -- identity + de-padding
      seq:add(con)
      seq:add(nn.CAddTable())
      return seq
   end
   local model = nn.Sequential()
   --model:add(skip(backend, ch, 64 - ch))
   model:add(SpatialConvolution(backend, ch, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(resblock(backend))
   model:add(resblock(backend))
   model:add(resblock(backend))
   model:add(resblock(backend))
   model:add(resblock(backend))
   model:add(resblock(backend))
   model:add(SpatialFullConvolution(backend, 64, 64, 4, 4, 2, 2, 2, 2))
   model:add(ReLU(backend))
   model:add(SpatialConvolution(backend, 64, ch, 3, 3, 1, 1, 0, 0))

   model:add(w2nn.InplaceClip01())
   --model:add(nn.View(-1):setNumInputDims(3))
   model.w2nn_arch_name = "srresnet_2x"
   model.w2nn_offset = 28
   model.w2nn_scale_factor = 2
   model.w2nn_resize = true
   model.w2nn_channels = ch

   --model:cuda()
   --print(model:forward(torch.Tensor(32, ch, 92, 92):uniform():cuda()):size())

   return model
end

-- large version of srresnet_2x. It's current best model but slow.
function srcnn.resnet_14l(backend, ch)
   local function resblock(backend, i, o)
      local seq = nn.Sequential()
      local con = nn.ConcatTable()
      local conv = nn.Sequential()
      conv:add(SpatialConvolution(backend, i, o, 3, 3, 1, 1, 0, 0))
      conv:add(nn.LeakyReLU(0.1, true))
      conv:add(SpatialConvolution(backend, o, o, 3, 3, 1, 1, 0, 0))
      conv:add(nn.LeakyReLU(0.1, true))
      con:add(conv)
      if i == o then
	 con:add(nn.SpatialZeroPadding(-2, -2, -2, -2)) -- identity + de-padding
      else
	 local seq = nn.Sequential()
	 seq:add(SpatialConvolution(backend, i, o, 1, 1, 1, 1, 0, 0))
	 seq:add(nn.SpatialZeroPadding(-2, -2, -2, -2))
	 con:add(seq)
      end
      seq:add(con)
      seq:add(nn.CAddTable())
      return seq
   end
   local model = nn.Sequential()
   model:add(SpatialConvolution(backend, ch, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(resblock(backend, 32, 64))
   model:add(resblock(backend, 64, 64))
   model:add(resblock(backend, 64, 128))
   model:add(resblock(backend, 128, 128))
   model:add(resblock(backend, 128, 256))
   model:add(resblock(backend, 256, 256))
   model:add(SpatialFullConvolution(backend, 256, ch, 4, 4, 2, 2, 3, 3):noBias())
   model:add(w2nn.InplaceClip01())
   model:add(nn.View(-1):setNumInputDims(3))
   model.w2nn_arch_name = "resnet_14l"
   model.w2nn_offset = 28
   model.w2nn_scale_factor = 2
   model.w2nn_resize = true
   model.w2nn_channels = ch

   --model:cuda()
   --print(model:forward(torch.Tensor(32, ch, 92, 92):uniform():cuda()):size())

   return model
end

-- for segmentation
function srcnn.fcn_v1(backend, ch)
   -- input_size = 120
   local model = nn.Sequential()
   --i = 120
   --model:cuda()
   --print(model:forward(torch.Tensor(32, ch, i, i):uniform():cuda()):size())

   model:add(SpatialConvolution(backend, ch, 32, 5, 5, 2, 2, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 32, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialMaxPooling(backend, 2, 2, 2, 2))

   model:add(SpatialConvolution(backend, 32, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialMaxPooling(backend, 2, 2, 2, 2))

   model:add(SpatialConvolution(backend, 64, 128, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 128, 128, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialMaxPooling(backend, 2, 2, 2, 2))

   model:add(SpatialConvolution(backend, 128, 256, 1, 1, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(nn.Dropout(0.5, false, true))

   model:add(SpatialFullConvolution(backend, 256, 128, 2, 2, 2, 2, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialFullConvolution(backend, 128, 128, 2, 2, 2, 2, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 128, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialFullConvolution(backend, 64, 64, 2, 2, 2, 2, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialFullConvolution(backend, 32, ch, 4, 4, 2, 2, 3, 3))

   model:add(w2nn.InplaceClip01())
   model:add(nn.View(-1):setNumInputDims(3))
   model.w2nn_arch_name = "fcn_v1"
   model.w2nn_offset = 36
   model.w2nn_scale_factor = 1
   model.w2nn_channels = ch
   model.w2nn_input_size = 120
   --model.w2nn_gcn = true
   
   return model
end
function srcnn.cupconv_14(backend, ch)
   local function skip(backend, n_input, n_output, pad)
      local con = nn.ConcatTable()
      local conv = nn.Sequential()
      local depad = nn.Sequential()
      conv:add(nn.SelectTable(1))
      conv:add(SpatialConvolution(backend, n_input, n_output, 3, 3, 1, 1, 0, 0))
      conv:add(nn.LeakyReLU(0.1, true))
      con:add(conv)
      con:add(nn.Identity())
      return con
   end
   local function concat(backend, n, ch, n_middle)
      local con = nn.ConcatTable()
      for i = 1, n do
	 local pad = i - 1
	 if i == 1 then
	    con:add(nn.Sequential():add(nn.SelectTable(i)))
	 else
	    local seq = nn.Sequential()
	    seq:add(nn.SelectTable(i))
	    if pad > 0 then
	       seq:add(nn.SpatialZeroPadding(-pad, -pad, -pad, -pad))
	    end
	    if i == n then
	       --seq:add(SpatialConvolution(backend, ch, n_middle, 1, 1, 1, 1, 0, 0))
	    else
	       seq:add(w2nn.GradWeight(0.025))
	       seq:add(SpatialConvolution(backend, n_middle, n_middle, 1, 1, 1, 1, 0, 0))
	    end
	    seq:add(nn.LeakyReLU(0.1, true))
	    con:add(seq)
	 end
      end
      return nn.Sequential():add(con):add(nn.JoinTable(2))
   end
   local model = nn.Sequential()
   local m = 64
   local n = 14

   model:add(nn.ConcatTable():add(nn.Identity()))
   for i = 1, n - 1 do
      if i == 1 then
	 model:add(skip(backend, ch, m))
      else
	 model:add(skip(backend, m, m))
      end
   end
   model:add(nn.FlattenTable())
   model:add(concat(backend, n, ch, m))
   model:add(SpatialFullConvolution(backend, m * (n - 1) + 3, ch, 4, 4, 2, 2, 3, 3):noBias())
   model:add(w2nn.InplaceClip01())
   model:add(nn.View(-1):setNumInputDims(3))

   model.w2nn_arch_name = "cupconv_14"
   model.w2nn_offset = 28
   model.w2nn_scale_factor = 2
   model.w2nn_channels = ch
   model.w2nn_resize = true

   return model
end

function srcnn.upconv_refine(backend, ch)
   local function block(backend, ch)
      local seq = nn.Sequential()
      local con = nn.ConcatTable()
      local res = nn.Sequential()
      local base = nn.Sequential()
      local refine = nn.Sequential()
      local aux_con = nn.ConcatTable()

      res:add(w2nn.GradWeight(0.1))
      res:add(SpatialConvolution(backend, ch, 32, 3, 3, 1, 1, 0, 0))
      res:add(nn.LeakyReLU(0.1, true))
      res:add(SpatialConvolution(backend, 32, 64, 3, 3, 1, 1, 0, 0))
      res:add(nn.LeakyReLU(0.1, true))
      res:add(SpatialConvolution(backend, 64, 128, 3, 3, 1, 1, 0, 0))
      res:add(nn.LeakyReLU(0.1, true))
      res:add(SpatialConvolution(backend, 128, ch, 3, 3, 1, 1, 0, 0):noBias())
      res:add(w2nn.InplaceClip01())
      res:add(nn.MulConstant(0.5))

      con:add(res)
      con:add(nn.Sequential():add(nn.SpatialZeroPadding(-4, -4, -4, -4)):add(nn.MulConstant(0.5)))

      -- main output
      refine:add(nn.CAddTable()) -- averaging
      refine:add(nn.View(-1):setNumInputDims(3))
      -- aux output
      base:add(nn.SelectTable(2))
      base:add(nn.MulConstant(2)) -- revert mul 0.5
      base:add(nn.View(-1):setNumInputDims(3))

      aux_con:add(refine)
      aux_con:add(base)

      seq:add(con)
      seq:add(aux_con)
      seq:add(w2nn.AuxiliaryLossTable(1))
      return seq
   end
   local model = nn.Sequential()
   model:add(SpatialConvolution(backend, ch, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 32, 32, 3, 3, 1, 1, 0, 0))
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
   model:add(w2nn.InplaceClip01())
   model:add(block(backend, ch))

   model.w2nn_arch_name = "upconv_refine"
   model.w2nn_offset = 18
   model.w2nn_scale_factor = 2
   model.w2nn_resize = true
   model.w2nn_channels = ch

   return model
end

-- cascade u-net
function srcnn.cunet_v1(backend, ch)
   function unet_branch(insert, backend, n_input, n_output, depad)
      local block = nn.Sequential()
      local pooling = SpatialConvolution(backend, n_input, n_input, 2, 2, 2, 2, 0, 0) -- downsampling
      --block:add(w2nn.Print())
      block:add(pooling)
      block:add(insert)
      block:add(SpatialFullConvolution(backend, n_output, n_output, 2, 2, 2, 2, 0, 0))-- upsampling
      local parallel = nn.ConcatTable(2)
      parallel:add(nn.SpatialZeroPadding(-depad, -depad, -depad, -depad))
      parallel:add(block)
      local model = nn.Sequential()
      model:add(parallel)
      model:add(nn.JoinTable(2))
      return model
   end
   function unet_conv(n_input, n_middle, n_output)
	local model = nn.Sequential()
	model:add(SpatialConvolution(backend, n_input, n_middle, 3, 3, 1, 1, 0, 0))
	model:add(nn.LeakyReLU(0.1, true))
	model:add(SpatialConvolution(backend, n_middle, n_output, 3, 3, 1, 1, 0, 0))
	return model
   end
   function unet(backend, ch, deconv)
      -- 
      local block1 = unet_conv(128, 256, 128)
      local block2 = nn.Sequential()
      block2:add(unet_conv(32, 64, 128))
      block2:add(unet_branch(block1, backend, 128, 128, 4))
      block2:add(unet_conv(128*2, 64, 32))
      local model = nn.Sequential()
      model:add(unet_conv(ch, 32, 32))
      model:add(unet_branch(block2, backend, 32, 32, 16))
      model:add(SpatialConvolution(backend, 64, 128, 3, 3, 1, 1, 0, 0))
      model:add(nn.LeakyReLU(0.1))
      if deconv then
	 model:add(SpatialFullConvolution(backend, 128, ch, 4, 4, 2, 2, 3, 3))
      else
	 model:add(SpatialConvolution(backend, 128, ch, 3, 3, 1, 1, 0, 0))
      end
      return model
   end
   local model = nn.Sequential()
   local con = nn.ConcatTable()
   local aux_con = nn.ConcatTable()

   model:add(unet(backend, ch, true))

   con:add(unet(backend, ch, false))
   con:add(nn.SpatialZeroPadding(-20, -20, -20, -20))

   aux_con:add(nn.Sequential():add(nn.CAddTable()):add(w2nn.InplaceClip01())) -- cascaded unet output
   aux_con:add(nn.Sequential():add(nn.SelectTable(2)):add(w2nn.InplaceClip01())) -- single unet output

   model:add(con)
   model:add(aux_con)
   model:add(w2nn.AuxiliaryLossTable(1)) -- auxiliary loss for single unet output
   
   model.w2nn_arch_name = "cunet_v1"
   model.w2nn_offset = 60
   model.w2nn_scale_factor = 2
   model.w2nn_channels = ch
   model.w2nn_resize = true
   -- 72, 128, 256 are valid
   --model.w2nn_input_size = 128

   return model
end

-- cascade u-net
function srcnn.cunet_v2(backend, ch)
   function unet_branch(insert, backend, n_input, n_output, depad)
      local block = nn.Sequential()
      local pooling = SpatialConvolution(backend, n_input, n_input, 2, 2, 2, 2, 0, 0) -- downsampling
      --block:add(w2nn.Print())
      block:add(pooling)
      block:add(insert)
      block:add(SpatialFullConvolution(backend, n_output, n_output, 2, 2, 2, 2, 0, 0))-- upsampling
      local parallel = nn.ConcatTable(2)
      parallel:add(nn.SpatialZeroPadding(-depad, -depad, -depad, -depad))
      parallel:add(block)
      local model = nn.Sequential()
      model:add(parallel)
      model:add(nn.CAddTable(2))
      return model
   end
   function unet_conv(n_input, n_middle, n_output)
	local model = nn.Sequential()
	model:add(SpatialConvolution(backend, n_input, n_middle, 3, 3, 1, 1, 0, 0))
	model:add(nn.LeakyReLU(0.1, true))
	model:add(SpatialConvolution(backend, n_middle, n_output, 3, 3, 1, 1, 0, 0))
	return model
   end
   -- res unet
   function unet(backend, ch, deconv)
      local block1 = unet_conv(128, 256, 128)
      local block2 = nn.Sequential()
      block2:add(unet_conv(64, 128, 128))
      block2:add(unet_branch(block1, backend, 128, 128, 4))
      block2:add(unet_conv(128, 128, 64))
      local model = nn.Sequential()
      model:add(nn.SpatialZeroPadding(-1, -1, -1, -1))
      model:add(SpatialConvolution(backend, ch, 64, 3, 3, 1, 1, 0, 0))
      model:add(unet_branch(block2, backend, 64, 64, 16))
      if deconv then
	 model:add(SpatialConvolution(backend, 64, 128, 3, 3, 1, 1, 0, 0))
	 model:add(nn.LeakyReLU(0.1))
	 model:add(SpatialFullConvolution(backend, 128, 64, 4, 4, 2, 2, 3, 3))
      else
	 model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
      end
      return model
   end
   local model = nn.Sequential()
   local con = nn.ConcatTable()
   local aux_con = nn.ConcatTable()

   model:add(unet(backend, ch, true))
   con:add(unet(backend, 64, false))
   con:add(nn.SpatialZeroPadding(-19, -19, -19, -19))

   model:add(con)
   model:add(nn.CAddTable())
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, 64, ch, 3, 3, 1, 1, 0, 0))
   
   model.w2nn_arch_name = "cunet_v2"
   model.w2nn_offset = 60
   model.w2nn_scale_factor = 2
   model.w2nn_channels = ch
   model.w2nn_resize = true
   -- 72, 128, 256 are valid
   --model.w2nn_input_size = 128

   return model
end
-- cascade u-net
function srcnn.cunet_v3(backend, ch)
   function unet_branch(insert, backend, n_input, n_output, depad)
      local block = nn.Sequential()
      local pooling = SpatialConvolution(backend, n_input, n_input, 2, 2, 2, 2, 0, 0) -- downsampling
      --block:add(w2nn.Print())
      block:add(pooling)
      block:add(insert)
      block:add(SpatialFullConvolution(backend, n_output, n_output, 2, 2, 2, 2, 0, 0))-- upsampling
      local parallel = nn.ConcatTable(2)
      parallel:add(nn.SpatialZeroPadding(-depad, -depad, -depad, -depad))
      parallel:add(block)
      local model = nn.Sequential()
      model:add(parallel)
      model:add(nn.CAddTable())
      return model
   end
   function unet_conv(n_input, n_middle, n_output)
	local model = nn.Sequential()
	model:add(SpatialConvolution(backend, n_input, n_middle, 3, 3, 1, 1, 0, 0))
	model:add(nn.LeakyReLU(0.1, true))
	model:add(SpatialConvolution(backend, n_middle, n_output, 3, 3, 1, 1, 0, 0))
	model:add(nn.LeakyReLU(0.1, true))
	return model
   end
   function unet(backend, ch, deconv)
      local block1 = unet_conv(128, 256, 128)
      local block2 = nn.Sequential()
      block2:add(unet_conv(64, 64, 128))
      block2:add(unet_branch(block1, backend, 128, 128, 4))
      block2:add(unet_conv(128, 64, 64))
      local model = nn.Sequential()
      model:add(unet_conv(ch, 32, 64))
      model:add(unet_branch(block2, backend, 64, 64, 16))
      if deconv then
	 model:add(SpatialConvolution(backend, 64, 128, 3, 3, 1, 1, 0, 0))
	 model:add(nn.LeakyReLU(0.1))
	 model:add(SpatialFullConvolution(backend, 128, 64, 4, 4, 2, 2, 3, 3))
      end
      return model
   end
   local model = nn.Sequential()
   local con = nn.ConcatTable()

   model:add(unet(backend, ch, true))
   model:add(nn.ConcatTable():add(unet(backend, 64, false)):add(nn.SpatialZeroPadding(-18, -18, -18, -18)))
   model:add(nn.CAddTable())
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU())
   model:add(SpatialConvolution(backend, 64, ch, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.InplaceClip01())
   
   model.w2nn_arch_name = "cunet_v3"
   model.w2nn_offset = 60
   model.w2nn_scale_factor = 2
   model.w2nn_channels = ch
   model.w2nn_resize = true
   -- 72, 128, 256 are valid
   --model.w2nn_input_size = 128

   return model
end
-- cascade u-net
function srcnn.cunet_v4(backend, ch)
   function upconv_3(backend, n_input, n_output)
      local model = nn.Sequential()
      model:add(SpatialConvolution(backend, n_input, 32, 3, 3, 1, 1, 0, 0))
      model:add(nn.LeakyReLU(0.1, true))
      model:add(SpatialConvolution(backend, 32, 32, 3, 3, 1, 1, 0, 0))
      model:add(nn.LeakyReLU(0.1, true))
      model:add(SpatialFullConvolution(backend, 32, n_output, 4, 4, 2, 2, 3, 3):noBias())
      return model
   end
   function unet_branch(insert, backend, n_input, n_output, depad)
      local block = nn.Sequential()
      local pooling = SpatialConvolution(backend, n_input, n_input, 2, 2, 2, 2, 0, 0) -- downsampling
      --block:add(w2nn.Print())
      block:add(pooling)
      block:add(insert)
      block:add(SpatialFullConvolution(backend, n_output, n_output, 2, 2, 2, 2, 0, 0))-- upsampling
      local parallel = nn.ConcatTable(2)
      parallel:add(nn.SpatialZeroPadding(-depad, -depad, -depad, -depad))
      parallel:add(block)
      local model = nn.Sequential()
      model:add(parallel)
      model:add(nn.CAddTable())
      return model
   end
   function unet_conv(n_input, n_middle, n_output)
	local model = nn.Sequential()
	model:add(SpatialConvolution(backend, n_input, n_middle, 3, 3, 1, 1, 0, 0))
	model:add(nn.LeakyReLU(0.1, true))
	model:add(SpatialConvolution(backend, n_middle, n_output, 3, 3, 1, 1, 0, 0))
	model:add(nn.LeakyReLU(0.1, true))
	return model
   end
   function unet(backend, ch)
      local block1 = unet_conv(128, 256, 128)
      local block2 = nn.Sequential()
      block2:add(unet_conv(64, 64, 128))
      block2:add(unet_branch(block1, backend, 128, 128, 4))
      block2:add(unet_conv(128, 64, 64))
      local model = nn.Sequential()
      model:add(SpatialConvolution(backend, ch, 64, 3, 3, 1, 1, 0, 0))
      model:add(nn.LeakyReLU(0.1, true))
      model:add(unet_branch(block2, backend, 64, 64, 16))
      return model
   end
   local model = nn.Sequential()
   local con = nn.ConcatTable()
   local aux_con = nn.ConcatTable()

   model:add(upconv_3(backend, ch, 64))

   con:add(unet(backend, 32))
   --con:add(nn.SpatialZeroPadding(-20, -20, -20, -20))

   aux_con:add(nn.Sequential():add(nn.CAddTable()):add(w2nn.InplaceClip01())) -- cascaded unet output
   aux_con:add(nn.Sequential():add(nn.SelectTable(2)):add(w2nn.InplaceClip01())) -- single output

   model:add(conn)
   model:add(nn.CAddTable())
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU())
   model:add(SpatialConvolution(backend, 64, ch, 3, 3, 1, 1, 0, 0))
   model:add(w2nn.InplaceClip01())
   model.w2nn_arch_name = "cunet_v3"
   model.w2nn_offset = 60
   model.w2nn_scale_factor = 2
   model.w2nn_channels = ch
   model.w2nn_resize = true
   -- 72, 128, 256 are valid
   --model.w2nn_input_size = 128

   return model
end

function srcnn.cunet_v6(backend, ch)
   function unet_branch(insert, backend, n_input, n_output, depad)
      local block = nn.Sequential()
      local pooling = SpatialConvolution(backend, n_input, n_input, 2, 2, 2, 2, 0, 0) -- downsampling
      --block:add(w2nn.Print())
      block:add(pooling)
      block:add(insert)
      block:add(SpatialFullConvolution(backend, n_output, n_output, 2, 2, 2, 2, 0, 0))-- upsampling
      local parallel = nn.ConcatTable(2)
      parallel:add(nn.SpatialZeroPadding(-depad, -depad, -depad, -depad))
      parallel:add(block)
      local model = nn.Sequential()
      model:add(parallel)
      model:add(nn.CAddTable())
      return model
   end
   function unet_conv(n_input, n_middle, n_output, se)
	local model = nn.Sequential()
	model:add(SpatialConvolution(backend, n_input, n_middle, 3, 3, 1, 1, 0, 0))
	model:add(nn.LeakyReLU(0.1, true))
	model:add(SpatialConvolution(backend, n_middle, n_output, 3, 3, 1, 1, 0, 0))
	model:add(nn.LeakyReLU(0.1, true))
	if se then
	   -- Squeeze and Excitation Networks
	   local con = nn.ConcatTable(2)
	   local attention = nn.Sequential()
	   attention:add(nn.SpatialAdaptiveAveragePooling(1, 1)) -- global average pooling
	   attention:add(SpatialConvolution(backend, n_output, math.floor(n_output / 4), 1, 1, 1, 1, 0, 0))
	   attention:add(nn.ReLU(true))
	   attention:add(SpatialConvolution(backend, math.floor(n_output / 4), n_output, 1, 1, 1, 1, 0, 0))
	   attention:add(nn.Sigmoid(true))
	   con:add(nn.Identity())                                                                          
	   con:add(attention)
	   model:add(con)
	   model:add(w2nn.ScaleTable())
	end
	return model
   end
   -- Residual U-Net
   function unet(backend, ch, deconv)
      local block1 = unet_conv(128, 256, 128, true)
      local block2 = nn.Sequential()
      block2:add(unet_conv(64, 64, 128, true))
      block2:add(unet_branch(block1, backend, 128, 128, 4))
      block2:add(unet_conv(128, 64, 64, true))
      local model = nn.Sequential()
      model:add(unet_conv(ch, 32, 64, false))
      model:add(unet_branch(block2, backend, 64, 64, 16))
      model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
      model:add(nn.LeakyReLU(0.1))
      if deconv then
	 model:add(SpatialFullConvolution(backend, 64, ch, 4, 4, 2, 2, 3, 3))
      else
	 model:add(SpatialConvolution(backend, 64, ch, 3, 3, 1, 1, 0, 0))
      end
      return model
   end
   local model = nn.Sequential()
   local con = nn.ConcatTable()
   local aux_con = nn.ConcatTable()

   model:add(unet(backend, ch, true))

   con:add(unet(backend, ch, false))
   con:add(nn.SpatialZeroPadding(-20, -20, -20, -20))

   aux_con:add(nn.Sequential():add(nn.CAddTable()):add(w2nn.InplaceClip01())) -- cascaded unet output
   aux_con:add(nn.Sequential():add(nn.SelectTable(2)):add(w2nn.InplaceClip01())) -- single unet output

   model:add(con)
   model:add(aux_con)
   model:add(w2nn.AuxiliaryLossTable(1)) -- auxiliary loss for single unet output
   
   model.w2nn_arch_name = "cunet_v6"
   model.w2nn_offset = 60
   model.w2nn_scale_factor = 2
   model.w2nn_channels = ch
   model.w2nn_resize = true
   -- 72, 128, 256 are valid
   --model.w2nn_input_size = 128

   return model
end

function srcnn.prog_net(backend, ch)
   function base_upscaler(backend, ch)
      local model = nn.Sequential()
      model:add(nn.SpatialZeroPadding(-11, -11, -11, -11))
      model:add(SpatialConvolution(backend, ch, 32, 3, 3, 1, 1, 0, 0))
      model:add(nn.LeakyReLU(0.1, true))
      model:add(SpatialConvolution(backend, 32, 64, 3, 3, 1, 1, 0, 0))
      model:add(nn.LeakyReLU(0.1, true))
      model:add(SpatialFullConvolution(backend, 64, ch, 4, 4, 2, 2, 3, 3):noBias())
      model:add(w2nn.InplaceClip01())
      return model
   end
   function block(backend, input, output)
      local con = nn.ConcatTable()
      local conv = nn.Sequential()
      local dil = nn.Sequential()
      local b = nn.Sequential()

      conv:add(SpatialConvolution(backend, input, output, 3, 3, 1, 1, 0, 0))
      conv:add(nn.SpatialZeroPadding(-5, -5, -5, -5))

      dil:add(SpatialDilatedConvolution(backend, input, output, 3, 3, 1, 1, 0, 0, 2, 2))
      dil:add(nn.LeakyReLU(0.1, true))
      dil:add(SpatialDilatedConvolution(backend, output, output, 3, 3, 1, 1, 0, 0, 4, 4))

      con:add(conv)
      con:add(dil)

      b:add(con)
      b:add(nn.CAddTable())
      b:add(nn.LeakyReLU(0.1, true))

      return b
   end
   function texture_upscaler(backend, ch)
      local model = nn.Sequential()
      model:add(w2nn.EdgeFilter(ch))
      model:add(SpatialConvolution(backend, ch * 8, 32, 1, 1, 1, 1, 0, 0))
      model:add(nn.LeakyReLU(0.1, true))
      model:add(block(backend, 32, 128))
      model:add(block(backend, 128, 256))
      model:add(SpatialFullConvolution(backend, 256, ch, 4, 4, 2, 2, 3, 3):noBias())
      return model
   end
   local model = nn.Sequential()
   local con = nn.ConcatTable()
   local aux = nn.ConcatTable()

   con:add(base_upscaler(backend, ch))
   con:add(texture_upscaler(backend, ch))

   aux:add(nn.Sequential():add(nn.CAddTable()):add(w2nn.InplaceClip01()):add(nn.View(-1):setNumInputDims(3)))
   aux:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.View(-1):setNumInputDims(3)))

   model:add(con)
   model:add(aux)
   model:add(w2nn.AuxiliaryLossTable(1))

   model.w2nn_arch_name = "prog_net"
   model.w2nn_offset = 28
   model.w2nn_scale_factor = 2
   model.w2nn_channels = ch
   model.w2nn_resize = true

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


--[[
local model = srcnn.fcn_v1("cunn", 3):cuda()
print(model:forward(torch.Tensor(1, 3, 108, 108):zero():cuda()):size())
print(model)
local model = srcnn.unet_refine("cunn", 3):cuda()
print(model)
print(model:forward(torch.Tensor(1, 3, 64, 64):zero():cuda()):size())
local model = srcnn.cupconv_14("cunn", 3):cuda()
print(model)
print(model:forward(torch.Tensor(1, 3, 64, 64):zero():cuda()):size())
os.exit()
local model = srcnn.cupconv_14("cunn", 3):cuda()
print(model)
print(model:forward(torch.Tensor(1, 3, 64, 64):zero():cuda()):size())
os.exit()

local model = srcnn.upconv_refine("cunn", 3):cuda()
print(model)
model:training()
print(model:forward(torch.Tensor(1, 3, 64, 64):zero():cuda()))
os.exit()

local model = srcnn.nw2("cunn", 3):cuda()
print(model)
model:training()
print(model:forward(torch.Tensor(1, 3, 64, 64):zero():cuda()))
os.exit()

local model = srcnn.prog_net("cunn", 3):cuda()
print(model)
model:training()
print(model:forward(torch.Tensor(1, 3, 128, 128):zero():cuda()))
os.exit()
local model = srcnn.double_unet("cunn", 3):cuda()
print(model)
model:training()
print(model:forward(torch.Tensor(1, 3, 144, 144):zero():cuda()))
os.exit()

local model = srcnn.cunet_v3("cunn", 3):cuda()
print(model)
model:training()
print(model:forward(torch.Tensor(1, 3, 144, 144):zero():cuda()):size())
os.exit()
local model = srcnn.cunet_v6("cunn", 3):cuda()
print(model)
model:training()
print(model:forward(torch.Tensor(1, 3, 144, 144):zero():cuda()))
os.exit()


--]]

return srcnn
