local RGBWeightedMSECriterion, parent = torch.class('mynn.RGBWeightedMSECriterion','nn.Criterion')

function RGBWeightedMSECriterion:__init(w)
   parent.__init(self)
   self.weight = w:clone()
   self.diff = torch.Tensor()
   self.loss = torch.Tensor()
end

function RGBWeightedMSECriterion:updateOutput(input, target)
   self.diff:resizeAs(input):copy(input)
   for i = 1, input:size(1) do
      self.diff[i]:add(-1, target[i]):cmul(self.weight)
   end
   self.loss:resizeAs(self.diff):copy(self.diff):cmul(self.diff)
   self.output = self.loss:mean()
   
   return self.output
end

function RGBWeightedMSECriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input):copy(self.diff)
   return self.gradInput
end

