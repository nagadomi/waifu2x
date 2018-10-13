local Print, parent = torch.class('w2nn.Print','nn.Module')

function Print:__init(id)
   parent.__init(self)
   self.id = id
end
function Print:updateOutput(input)
   print("----", self.id)
   print(input:size())
   self.output:resizeAs(input)
   self.output:copy(input)
   return self.output
end
function Print:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(GradOutput)
   return self.gradInput
end
