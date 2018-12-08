local Clip01, parent = torch.class("w2nn.InplaceClip01", "nn.Module")

function Clip01:__init()
   parent.__init(self)
end
function Clip01:updateOutput(input)
   self.output:set(input:clamp(0, 1))
   return self.output
end
function Clip01:updateGradInput(input, gradOutput)
   self.gradInput:set(gradOutput)
   return self.gradInput
end
