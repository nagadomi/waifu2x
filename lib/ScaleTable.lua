local ScaleTable, parent = torch.class("w2nn.ScaleTable", "nn.Module")

function ScaleTable:__init()
   parent.__init(self)
   self.gradInput = {}
   self.grad_tmp = torch.Tensor()
   self.scale = torch.Tensor()
end
function ScaleTable:updateOutput(input)
   assert(#input == 2)
   assert(input[1]:size(2) == input[2]:size(2))

   self.scale:resizeAs(input[1]):expandAs(input[2], input[1])
   self.output:resizeAs(self.scale):copy(self.scale)
   self.output:cmul(input[1])
   return self.output
end

function ScaleTable:updateGradInput(input, gradOutput)
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[1]:resizeAs(input[1]):copy(gradOutput)
   self.gradInput[1]:cmul(self.scale)

   self.grad_tmp:resizeAs(input[1]):copy(gradOutput)
   self.grad_tmp:cmul(input[1])
   self.gradInput[2] = self.gradInput[2] or input[2].new()
   self.gradInput[2]:resizeAs(input[2]):sum(self.grad_tmp:reshape(self.grad_tmp:size(1), self.grad_tmp:size(2), self.grad_tmp:size(3) * self.grad_tmp:size(4)), 3):resizeAs(input[2])

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end
function ScaleTable:clearState()
   nn.utils.clear(self, {'grad_tmp','scale'})
   return parent:clearState()
end
