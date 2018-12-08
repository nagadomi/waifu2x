-- RandomBinaryConvolution.lua
-- from https://github.com/juefeix/lbcnn.torch

local THNN = require 'nn.THNN'
local RandomBinaryConvolution, parent = torch.class('w2nn.RandomBinaryConvolution', 'nn.SpatialConvolution')

function RandomBinaryConvolution:__init(nInputPlane, nOutputPlane, kW, kH, kSparsity)
   self.kSparsity = kSparsity or 0.9
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, 1, 1, 0, 0)
   self:reset()
end
function RandomBinaryConvolution:reset()
   local numElements = self.nInputPlane*self.nOutputPlane*self.kW*self.kH
   self.weight:fill(0)
   self.weight = torch.reshape(self.weight,numElements)
   local index = torch.Tensor(torch.floor(self.kSparsity*numElements)):random(numElements)
   for i = 1, index:numel() do
      self.weight[index[i]] = torch.bernoulli(0.5)*2-1
   end
   self.weight = torch.reshape(self.weight,self.nOutputPlane,self.nInputPlane,self.kW,self.kH)
   self.bias = nil
   self.gradBias = nil	
   self.gradWeight:fill(0)
end
function RandomBinaryConvolution:accGradParameters(input, gradOutput, scale)
end
function RandomBinaryConvolution:updateParameters(learningRate)
end
