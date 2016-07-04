local ClippedMSECriterion, parent = torch.class('w2nn.ClippedMSECriterion','nn.Criterion')

function ClippedMSECriterion:__init(min, max)
   parent.__init(self)
   self.min = min
   self.max = max
   self.diff = torch.Tensor()
end
function ClippedMSECriterion:updateOutput(input, target)
   self.diff:resizeAs(input):copy(input)
   self.diff:clamp(self.min, self.max)
   self.diff:add(-1, target)
   self.output = self.diff:pow(2):sum() / input:nElement()
   return self.output
end
function ClippedMSECriterion:updateGradInput(input, target)
   local norm = 1.0 / input:nElement()
   self.gradInput:resizeAs(self.diff):copy(self.diff):mul(norm)
   return self.gradInput 
end
