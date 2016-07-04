-- ref: https://en.wikipedia.org/wiki/Huber_loss
local ClippedWeightedHuberCriterion, parent = torch.class('w2nn.ClippedWeightedHuberCriterion','nn.Criterion')

function ClippedWeightedHuberCriterion:__init(w, gamma, clip)
   parent.__init(self)
   self.clip = clip
   self.gamma = gamma or 1.0
   self.weight = w:clone()
   self.diff = torch.Tensor()
   self.diff_abs = torch.Tensor()
   --self.outlier_rate = 0.0
   self.square_loss_buff = torch.Tensor()
   self.linear_loss_buff = torch.Tensor()
end
function ClippedWeightedHuberCriterion:updateOutput(input, target)
   self.diff:resizeAs(input):copy(input)
   self.diff:clamp(self.clip[1], self.clip[2])
   for i = 1, input:size(1) do
      self.diff[i]:add(-1, target[i]):cmul(self.weight)
   end
   self.diff_abs:resizeAs(self.diff):copy(self.diff):abs()
   
   local square_targets = self.diff[torch.lt(self.diff_abs, self.gamma)]
   local linear_targets = self.diff[torch.ge(self.diff_abs, self.gamma)]
   local square_loss = self.square_loss_buff:resizeAs(square_targets):copy(square_targets):pow(2.0):mul(0.5):sum()
   local linear_loss = self.linear_loss_buff:resizeAs(linear_targets):copy(linear_targets):abs():add(-0.5 * self.gamma):mul(self.gamma):sum()

   --self.outlier_rate = linear_targets:nElement() / input:nElement()
   self.output = (square_loss + linear_loss) / input:nElement()
   return self.output
end
function ClippedWeightedHuberCriterion:updateGradInput(input, target)
   local norm = 1.0 / input:nElement()
   self.gradInput:resizeAs(self.diff):copy(self.diff):mul(norm)
   local outlier = torch.ge(self.diff_abs, self.gamma)
   self.gradInput[outlier] = torch.sign(self.diff[outlier]) * self.gamma * norm
   return self.gradInput 
end
