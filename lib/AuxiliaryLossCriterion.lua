require 'nn'
local AuxiliaryLossCriterion, parent = torch.class('w2nn.AuxiliaryLossCriterion','nn.Criterion')

function AuxiliaryLossCriterion:__init(base_criterion, args)
   parent.__init(self)
   self.base_criterion = base_criterion
   self.args = args
   self.criterions = {}
   self.gradInput = {}
   self.sizeAverage = false
end
function AuxiliaryLossCriterion:updateOutput(input, target)
   local sum_output = 0
   if type(input) == "table" then
      -- model:training()
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
	 local output = self.criterions[i]:updateOutput(input[i], target)
	 sum_output = sum_output + output
      end
      self.output = sum_output / #input
   else
      -- model:evaluate()
      if self.criterions[1] == nil then
	 if self.args ~= nil then
	    self.criterions[1] = self.base_criterion(table.unpack(self.args))
	 else
	    self.criterions[1] = self.base_criterion()
	 end
	 self.criterions[1].sizeAverage = self.sizeAverage()
	 if input:type() == "torch.CudaTensor" then
	    self.criterions[1]:cuda()
	 end
      end
      self.output = self.criterions[1]:updateOutput(input, target)
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
