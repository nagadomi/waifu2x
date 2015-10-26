require 'torch'
require 'nn'

local function load_cuda()
   require 'cutorch'
   require 'cunn'
end
local function load_cudnn()
   require 'cudnn'
   --cudnn.fastest = true
end

if pcall(load_cuda) then
else
end
if pcall(load_cudnn) then
end
