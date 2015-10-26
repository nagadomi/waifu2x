local function load_cuda()
   require 'nn'
   require 'cunn'
end
local function load_cudnn()
   require 'cudnn'
   --cudnn.fastest = true
end

if pcall(load_cuda) then
else
   --[[ TODO: fakecuda does not work.
      
   io.stderr:write("use FakeCUDA; if you have NVIDIA GPU, Please install cutorch and cunn. FakeCuda will be extremely slow.\n")
   require 'torch'
   require 'nn'
   require('fakecuda').init(true)
   --]]
end
if pcall(load_cudnn) then
end
