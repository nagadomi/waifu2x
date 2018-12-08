local GradWeight, parent = torch.class('w2nn.GradWeight', 'nn.Module')

function GradWeight:__init(constant_scalar)
  parent.__init(self)
  assert(type(constant_scalar) == 'number', 'input is not scalar!')
  self.constant_scalar = constant_scalar
end

function GradWeight:updateOutput(input)
  self.output:resizeAs(input)
  self.output:copy(input)
  return self.output
end

function GradWeight:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput)
   self.gradInput:copy(gradOutput)
   self.gradInput:mul(self.constant_scalar)
   return self.gradInput
end
