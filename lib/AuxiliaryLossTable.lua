require 'nn'
local AuxiliaryLossTable, parent = torch.class('w2nn.AuxiliaryLossTable', 'nn.Module')

function AuxiliaryLossTable:__init(i)
   parent.__init(self)
   self.i = i or 1
   self.gradInput = {}
   self.output_table = {}
   self.output_tensor = torch.Tensor()
end

function AuxiliaryLossTable:updateOutput(input)
   if self.train then
      for i=1,#input do
	 self.output_table[i] = self.output_table[i] or input[1].new()
	 self.output_table[i]:resizeAs(input[i]):copy(input[i])
      end
      for i=#input+1, #self.output_table do
	 self.output_table[i] = nil
      end
      self.output = self.output_table
   else
      self.output_tensor:resizeAs(input[1]):copy(input[1])
      self.output_tensor:copy(input[self.i])
      self.output = self.output_tensor
   end
   return self.output
end

function AuxiliaryLossTable:updateGradInput(input, gradOutput)
   for i=1,#input do
      self.gradInput[i] = self.gradInput[i] or input[1].new()
      self.gradInput[i]:resizeAs(input[i]):copy(gradOutput[i])
   end
   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end
function AuxiliaryLossTable:clearState()
   self.gradInput = {}
   self.output_table = {}
   nn.utils.clear(self, 'output_tensor')
   return parent:clearState()
end
