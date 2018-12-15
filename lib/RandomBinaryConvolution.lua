-- RandomBinaryConvolution.lua
-- from https://github.com/juefeix/lbcnn.torch
--[[
MIT License

Copyright (c) 2017 Felix Juefei Xu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--]]

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
