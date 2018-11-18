local PrintTable, parent = torch.class('w2nn.PrintTable','nn.Module')

function PrintTable:__init(id)
   parent.__init(self)
   self.id = id
end
function PrintTable:updateOutput(input)
   print("----", self.id)
   print(input)
   self.output = input
   return self.output
end
function PrintTable:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end
