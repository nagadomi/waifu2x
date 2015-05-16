require 'optim'
require 'cutorch'
require 'xlua'

local function minibatch_sgd(model, criterion,
			     train_x,
			     config, transformer,
			     input_size, target_size)
   local parameters, gradParameters = model:getParameters()
   config = config or {}
   local sum_loss = 0
   local count_loss = 0
   local batch_size = config.xBatchSize or 32
   local shuffle = torch.randperm(#train_x)
   local c = 1
   local inputs = torch.Tensor(batch_size,
			       input_size[1], input_size[2], input_size[3]):cuda()
   local targets = torch.Tensor(batch_size,
				target_size[1] * target_size[2] * target_size[3]):cuda()
   local inputs_tmp = torch.Tensor(batch_size,
			       input_size[1], input_size[2], input_size[3])
   local targets_tmp = torch.Tensor(batch_size,
				    target_size[1] * target_size[2] * target_size[3])
   
   for t = 1, #train_x, batch_size do
      if t + batch_size > #train_x then
	 break
      end
      xlua.progress(t, #train_x)
      for i = 1, batch_size do
	 local x, y = transformer(train_x[shuffle[t + i - 1]])
         inputs_tmp[i]:copy(x)
	 targets_tmp[i]:copy(y)
      end
      inputs:copy(inputs_tmp)
      targets:copy(targets_tmp)
      
      local feval = function(x)
	 if x ~= parameters then
	    parameters:copy(x)
	 end
	 gradParameters:zero()
	 local output = model:forward(inputs)
	 local f = criterion:forward(output, targets)
	 sum_loss = sum_loss + f
	 count_loss = count_loss + 1
	 model:backward(inputs, criterion:backward(output, targets))
	 return f, gradParameters
      end
      -- must use Adam!!
      optim.adam(feval, parameters, config)
      
      c = c + 1
      if c % 10 == 0 then
	 collectgarbage()
      end
   end
   xlua.progress(#train_x, #train_x)
   
   return { mse = sum_loss / count_loss}
end

return minibatch_sgd
