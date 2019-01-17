require 'nn'
local AuxiliaryLossCriterion, parent = torch.class('w2nn.AuxiliaryLossCriterion','nn.Criterion')

function AuxiliaryLossCriterion:__init(base_criterion, args)
   parent.__init(self)
   self.base_criterion = base_criterion
   self.args = args
   self.gradInput = {}
   self.sizeAverage = false
   self.criterions = {}
   if self.base_criterion.has_instance_loss then
      self.instance_loss = {}
   end
end
function AuxiliaryLossCriterion:updateOutput(input, target)
   local sum_output = 0
   if type(input) == "table" then
      -- model:training()
      self.output = 0
      for i = 1, #input do
	 if self.criterions[i] == nil then
	    if self.args ~= nil then
	       self.criterions[i] = self.base_criterion(table.unpack(self.args))
	    else
	       self.criterions[i] = self.base_criterion()
	    end
	    self.criterions[i].sizeAverage = self.sizeAverage
	    if input[i]:type() == "torch.CudaTensor" then
	       self.criterions[i]:cuda()
	    end
	 end
	 self.output = self.output + self.criterions[i]:updateOutput(input[i], target) / #input

	 if self.instance_loss then
	    local batch_size = #self.criterions[i].instance_loss
	    local scale = 1.0 / #input
	    if i == 1 then
	       for j = 1, batch_size do
		  self.instance_loss[j] = self.criterions[i].instance_loss[j] * scale
	       end
	    else
	       for j = 1, batch_size do
		  self.instance_loss[j] = self.instance_loss[j] + self.criterions[i].instance_loss[j] * scale
	       end
	    end
	 end
      end
   else
      -- model:evaluate()
      if self.criterions[1] == nil then
	 if self.args ~= nil then
	    self.criterions[1] = self.base_criterion(table.unpack(self.args))
	 else
	    self.criterions[1] = self.base_criterion()
	 end
	 self.criterions[1].sizeAverage = self.sizeAverage
	 if input:type() == "torch.CudaTensor" then
	    self.criterions[1]:cuda()
	 end
      end
      self.output = self.criterions[1]:updateOutput(input, target)
      if self.instance_loss then
	 local batch_size = #self.criterions[1].instance_loss
	 for j = 1, batch_size do
	    self.instance_loss[j] = self.criterions[1].instance_loss[j]
	 end
      end
   end
   return self.output
end

function AuxiliaryLossCriterion:updateGradInput(input, target)
   for i=1,#input do
      local gradInput = self.criterions[i]:updateGradInput(input[i], target)
      self.gradInput[i] = self.gradInput[i] or gradInput.new()
      self.gradInput[i]:resizeAs(gradInput):copy(gradInput)
   end
   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end
   return self.gradInput
end
