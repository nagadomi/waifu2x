local ClippedMSECriterion, parent = torch.class('w2nn.ClippedMSECriterion','nn.Criterion')

ClippedMSECriterion.has_instance_loss = true
function ClippedMSECriterion:__init(min, max)
   parent.__init(self)
   self.min = min or 0
   self.max = max or 1
   self.diff = torch.Tensor()
   self.diff_pow2 = torch.Tensor()
   self.instance_loss = {}
end
function ClippedMSECriterion:updateOutput(input, target)
   self.diff:resizeAs(input):copy(input)
   self.diff:clamp(self.min, self.max)
   self.diff:add(-1, target)
   self.diff_pow2:resizeAs(self.diff):copy(self.diff):pow(2)
   self.instance_loss = {}
   self.output = 0
   local scale = 1.0 / input:size(1)
   for i = 1, input:size(1) do
      local instance_loss = self.diff_pow2[i]:sum() / self.diff_pow2[i]:nElement()
      self.instance_loss[i] = instance_loss
      self.output = self.output + instance_loss
   end
   return self.output / input:size(1)
end
function ClippedMSECriterion:updateGradInput(input, target)
   local norm = 1.0 / input:nElement()
   self.gradInput:resizeAs(self.diff):copy(self.diff):mul(norm)
   return self.gradInput 
end
