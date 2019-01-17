require 'w2nn'

-- ref: https://arxiv.org/abs/1502.01852
-- ref: https://arxiv.org/abs/1501.00092
-- ref: https://arxiv.org/abs/1709.01507
-- ref: https://arxiv.org/abs/1505.04597
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

local function Sigmoid(backend)
   if backend == "cunn" then
      return nn.Sigmoid(true)
   elseif backend == "cudnn" then
      return cudnn.Sigmoid(true)
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

local function GlobalAveragePooling(n_output)
   local gap = nn.Sequential()
   gap:add(nn.Mean(-1, -1)):add(nn.Mean(-1, -1))
   gap:add(nn.View(-1, n_output, 1, 1))
   return gap
end
srcnn.GlobalAveragePooling = GlobalAveragePooling

-- Squeeze and Excitation Block
local function SEBlock(backend, n_output, r)
   local con = nn.ConcatTable(2)
   local attention = nn.Sequential()
   local n_mid = math.floor(n_output / r)
   attention:add(GlobalAveragePooling(n_output))
   attention:add(SpatialConvolution(backend, n_output, n_mid, 1, 1, 1, 1, 0, 0))
   attention:add(nn.ReLU(true))
   attention:add(SpatialConvolution(backend, n_mid, n_output, 1, 1, 1, 1, 0, 0))
   attention:add(nn.Sigmoid(true)) -- don't use cudnn sigmoid 
   con:add(nn.Identity())
   con:add(attention)
   return con
end
local function SpatialSEBlock(backend, ave_size, n_output, r)
   local con = nn.ConcatTable(2)
   local attention = nn.Sequential()
   local n_mid = math.floor(n_output / r)
   attention:add(SpatialAveragePooling(backend, ave_size, ave_size, ave_size, ave_size))
   attention:add(SpatialConvolution(backend, n_output, n_mid, 1, 1, 1, 1, 0, 0))
   attention:add(nn.ReLU(true))
   attention:add(SpatialConvolution(backend, n_mid, n_output, 1, 1, 1, 1, 0, 0))
   attention:add(nn.Sigmoid(true))
   attention:add(nn.SpatialUpSamplingNearest(ave_size, ave_size))
   con:add(nn.Identity())
   con:add(attention)
   return con
end
local function ResBlock(backend, i, o)
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
local function ResBlockSE(backend, i, o)
   local seq = nn.Sequential()
   local con = nn.ConcatTable()
   local conv = nn.Sequential()
   conv:add(SpatialConvolution(backend, i, o, 3, 3, 1, 1, 0, 0))
   conv:add(nn.LeakyReLU(0.1, true))
   conv:add(SpatialConvolution(backend, o, o, 3, 3, 1, 1, 0, 0))
   conv:add(nn.LeakyReLU(0.1, true))
   conv:add(SEBlock(backend, o, 8))
   conv:add(w2nn.ScaleTable())
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
local function ResGroup(backend, n, n_output)
   local seq = nn.Sequential()
   local res = nn.Sequential()
   local con = nn.ConcatTable(2)
   local depad = -2 * n
   for i = 1, n do
      res:add(ResBlock(backend, n_output, n_output))
   end
   con:add(res)
   con:add(nn.SpatialZeroPadding(depad, depad, depad, depad))
   seq:add(con)
   seq:add(nn.CAddTable())
   return seq
end
local function ResGroupSE(backend, n, n_output)
   local seq = nn.Sequential()
   local res = nn.Sequential()
   local con = nn.ConcatTable(2)
   local depad = -2 * n
   for i = 1, n do
      res:add(ResBlockSE(backend, n_output, n_output))
   end
   con:add(res)
   con:add(nn.SpatialZeroPadding(depad, depad, depad, depad))
   seq:add(con)
   seq:add(nn.CAddTable())
   return seq
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
   model:add(w2nn.InplaceClip01())
   model:add(nn.View(-1):setNumInputDims(3))

   model.w2nn_arch_name = "vgg_7"
   model.w2nn_offset = 7
   model.w2nn_scale_factor = 1
   model.w2nn_channels = ch
   
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

   return model
end

function srcnn.resnet_14l(backend, ch)
   local model = nn.Sequential()
   model:add(SpatialConvolution(backend, ch, 32, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(ResBlock(backend, 32, 64))
   model:add(ResBlock(backend, 64, 64))
   model:add(ResBlock(backend, 64, 128))
   model:add(ResBlock(backend, 128, 128))
   model:add(ResBlock(backend, 128, 256))
   model:add(ResBlock(backend, 256, 256))
   model:add(SpatialFullConvolution(backend, 256, ch, 4, 4, 2, 2, 3, 3):noBias())
   model:add(w2nn.InplaceClip01())
   model:add(nn.View(-1):setNumInputDims(3))
   model.w2nn_arch_name = "resnet_14l"
   model.w2nn_offset = 28
   model.w2nn_scale_factor = 2
   model.w2nn_resize = true
   model.w2nn_channels = ch

   return model
end

-- ResNet with SEBlock for fast conversion
function srcnn.upresnet_s(backend, ch)
   local model = nn.Sequential()
   model:add(SpatialConvolution(backend, ch, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(ResGroupSE(backend, 3, 64))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialFullConvolution(backend, 64, ch, 4, 4, 2, 2, 3, 3):noBias())
   model:add(w2nn.InplaceClip01())
   model.w2nn_arch_name = "upresnet_s"
   model.w2nn_offset = 18
   model.w2nn_scale_factor = 2
   model.w2nn_resize = true
   model.w2nn_channels = ch

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

-- Cascaded Residual U-Net with SEBlock

-- unet utils adapted from https://gist.github.com/toshi-k/ca75e614f1ac12fa44f62014ac1d6465
local function unet_conv(backend, n_input, n_middle, n_output, se)
   local model = nn.Sequential()
   model:add(SpatialConvolution(backend, n_input, n_middle, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   model:add(SpatialConvolution(backend, n_middle, n_output, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1, true))
   if se then
      model:add(SEBlock(backend, n_output, 8))
      model:add(w2nn.ScaleTable())
   end
   return model
end
local function unet_branch(backend, insert, backend, n_input, n_output, depad)
   local block = nn.Sequential()
   local con = nn.ConcatTable(2)
   local model = nn.Sequential()
   
   block:add(SpatialConvolution(backend, n_input, n_input, 2, 2, 2, 2, 0, 0))-- downsampling
   block:add(nn.LeakyReLU(0.1, true))
   block:add(insert)
   block:add(SpatialFullConvolution(backend, n_output, n_output, 2, 2, 2, 2, 0, 0))-- upsampling
   block:add(nn.LeakyReLU(0.1, true))
   con:add(block)
   con:add(nn.SpatialZeroPadding(-depad, -depad, -depad, -depad))
   model:add(con)
   model:add(nn.CAddTable())
   return model
end
local function cunet_unet1(backend, ch, deconv)
   local block1 = unet_conv(backend, 64, 128, 64, true)
   local model = nn.Sequential()
   model:add(unet_conv(backend, ch, 32, 64, false))
   model:add(unet_branch(backend, block1, backend, 64, 64, 4))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   if deconv then
	 model:add(SpatialFullConvolution(backend, 64, ch, 4, 4, 2, 2, 3, 3))
   else
      model:add(SpatialConvolution(backend, 64, ch, 3, 3, 1, 1, 0, 0))
   end
   return model
end
local function cunet_unet2(backend, ch, deconv)
   local block1 = unet_conv(backend, 128, 256, 128, true)
   local block2 = nn.Sequential()
   block2:add(unet_conv(backend, 64, 64, 128, true))
   block2:add(unet_branch(backend, block1, backend, 128, 128, 4))
   block2:add(unet_conv(backend, 128, 64, 64, true))
   local model = nn.Sequential()
   model:add(unet_conv(backend, ch, 32, 64, false))
   model:add(unet_branch(backend, block2, backend, 64, 64, 16))
   model:add(SpatialConvolution(backend, 64, 64, 3, 3, 1, 1, 0, 0))
   model:add(nn.LeakyReLU(0.1))
   if deconv then
      model:add(SpatialFullConvolution(backend, 64, ch, 4, 4, 2, 2, 3, 3))
   else
      model:add(SpatialConvolution(backend, 64, ch, 3, 3, 1, 1, 0, 0))
   end
   return model
end
-- 2x
function srcnn.upcunet(backend, ch)
   local model = nn.Sequential()
   local con = nn.ConcatTable()
   local aux_con = nn.ConcatTable()

   -- 2 cascade
   model:add(cunet_unet1(backend, ch, true))
   con:add(cunet_unet2(backend, ch, false))
   con:add(nn.SpatialZeroPadding(-20, -20, -20, -20))

   aux_con:add(nn.Sequential():add(nn.CAddTable()):add(w2nn.InplaceClip01())) -- cascaded unet output
   aux_con:add(nn.Sequential():add(nn.SelectTable(2)):add(w2nn.InplaceClip01())) -- single unet output

   model:add(con)
   model:add(aux_con)
   model:add(w2nn.AuxiliaryLossTable(1)) -- auxiliary loss for single unet output

   model.w2nn_arch_name = "upcunet"
   model.w2nn_offset = 36
   model.w2nn_scale_factor = 2
   model.w2nn_channels = ch
   model.w2nn_resize = true
   model.w2nn_valid_input_size = {}
   for i = 76, 512, 4 do
      table.insert(model.w2nn_valid_input_size, i)
   end

   return model
end
-- 1x
function srcnn.cunet(backend, ch)
   local model = nn.Sequential()
   local con = nn.ConcatTable()
   local aux_con = nn.ConcatTable()

   -- 2 cascade
   model:add(cunet_unet1(backend, ch, false))
   con:add(cunet_unet2(backend, ch, false))
   con:add(nn.SpatialZeroPadding(-20, -20, -20, -20))

   aux_con:add(nn.Sequential():add(nn.CAddTable()):add(w2nn.InplaceClip01())) -- cascaded unet output
   aux_con:add(nn.Sequential():add(nn.SelectTable(2)):add(w2nn.InplaceClip01())) -- single unet output

   model:add(con)
   model:add(aux_con)
   model:add(w2nn.AuxiliaryLossTable(1)) -- auxiliary loss for single unet output
   
   model.w2nn_arch_name = "cunet"
   model.w2nn_offset = 28
   model.w2nn_scale_factor = 1
   model.w2nn_channels = ch
   model.w2nn_resize = false
   model.w2nn_valid_input_size = {}
   for i = 100, 512, 4 do
      table.insert(model.w2nn_valid_input_size, i)
   end

   return model
end

local function bench()
   local sys = require 'sys'
   cudnn.benchmark = true
   local model = nil
   local arch = {"upconv_7", "upcunet", "vgg_7", "cunet"}
   local backend = "cudnn"
   local ch = 3
   local batch_size = 1
   local output_size = 256
   for k = 1, #arch do
      model = srcnn[arch[k]](backend, ch):cuda()
      model:evaluate()
      local dummy = nil
      local crop_size = nil
      if model.w2nn_resize then
	 crop_size = (output_size + model.w2nn_offset * 2) / 2
      else
	 crop_size = (output_size + model.w2nn_offset * 2)
      end
      local dummy = torch.Tensor(batch_size, ch, output_size, output_size):zero():cuda()

      print(arch[k], output_size, crop_size)
      -- warn
      for i = 1, 4 do
	 local x = torch.Tensor(batch_size, ch, crop_size, crop_size):uniform():cuda()
	 model:forward(x)
      end
      t = sys.clock()
      for i = 1, 10 do
	 local x = torch.Tensor(batch_size, ch, crop_size, crop_size):uniform():cuda()
	 local z = model:forward(x)
	 dummy:add(z)
      end
      print(arch[k], sys.clock() - t)
      model:clearState()
   end
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
local model = srcnn.resnet_s("cunn", 3):cuda()
print(model)
model:training()
print(model:forward(torch.Tensor(1, 3, 128, 128):zero():cuda()):size())
bench()
os.exit()
--]]
return srcnn
