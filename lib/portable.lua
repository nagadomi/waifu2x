local function load_cuda()
   require 'cunn'
end

if pcall(load_cuda) then
   require 'cunn'
else
   --[[ TODO: fakecuda does not work.
      
   io.stderr:write("use FakeCUDA; if you have NVIDIA GPU, Please install cutorch and cunn. FakeCuda will be extremely slow.\n")
   require 'torch'
   require 'nn'
   require('fakecuda').init(true)
   --]]
end
