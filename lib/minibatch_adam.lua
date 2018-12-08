require 'optim'
require 'cutorch'
require 'xlua'

local function minibatch_adam(model, criterion, eval_metric,
			      train_x, train_y,
			      config)
   local parameters, gradParameters = model:getParameters()
   config = config or {}
   if config.xEvalCount == nil then
      config.xEvalCount = 0
      config.learningRate = config.xLearningRate
   end
   local sum_psnr = 0
   local sum_loss = 0
   local sum_eval = 0
   local count_loss = 0
   local batch_size = config.xBatchSize or 32
   local shuffle = torch.randperm(train_x:size(1))
   local c = 1
   local inputs_tmp = torch.Tensor(batch_size,
				   train_x:size(2), train_x:size(3), train_x:size(4)):zero()
   local targets_tmp = torch.Tensor(batch_size,
				    train_y:size(2)):zero()
   local inputs = inputs_tmp:clone():cuda()
   local targets = targets_tmp:clone():cuda()
   local instance_loss = torch.Tensor(train_x:size(1)):zero()

   print("## update")
   for t = 1, train_x:size(1), batch_size do
      if t + batch_size -1 > train_x:size(1) then
	 break
      end
      for i = 1, batch_size do
         inputs_tmp[i]:copy(train_x[shuffle[t + i - 1]])
	 targets_tmp[i]:copy(train_y[shuffle[t + i - 1]])
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
	 local se = eval_metric:forward(output, targets)
	 if config.xInstanceLoss then
	    assert(eval_metric.instance_loss, "eval metric does not support instalce_loss")
	    for i = 1, #eval_metric.instance_loss do
	       instance_loss[shuffle[t + i - 1]] = eval_metric.instance_loss[i]
	    end
	 end
	 sum_psnr = sum_psnr + (10 * math.log10(1 / (se + 1.0e-6)))
	 sum_eval = sum_eval + se
	 sum_loss = sum_loss + f
	 count_loss = count_loss + 1
	 model:backward(inputs, criterion:backward(output, targets))
	 return f, gradParameters
      end
      optim.adam(feval, parameters, config)
      config.xEvalCount = config.xEvalCount + batch_size
      config.learningRate = config.xLearningRate / (1 + config.xEvalCount * config.xLearningRateDecay)
      c = c + 1
      if c % 50 == 0 then
	 collectgarbage()
	 xlua.progress(t, train_x:size(1))
      end
   end
   xlua.progress(train_x:size(1), train_x:size(1))
   return { loss = sum_loss / count_loss, MSE = sum_eval / count_loss, PSNR = sum_psnr / count_loss}, instance_loss
end

return minibatch_adam
