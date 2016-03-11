local PSNRCriterion, parent = torch.class('w2nn.PSNRCriterion','nn.Criterion')

function PSNRCriterion:__init()
   parent.__init(self)
   self.image = torch.Tensor()
   self.diff = torch.Tensor()
end
function PSNRCriterion:updateOutput(input, target)
   self.image:resizeAs(input):copy(input)
   self.image:clamp(0.0, 1.0)
   self.diff:resizeAs(self.image):copy(self.image)
   
   local mse = self.diff:add(-1, target):pow(2):mean()
   self.output = 10 * math.log10(1.0 / mse)
   return self.output
end
function PSNRCriterion:updateGradInput(input, target)
   error("PSNRCriterion does not support backward")
end
