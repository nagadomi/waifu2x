-- ref: https://en.wikipedia.org/wiki/L1_loss
local L1Criterion, parent = torch.class('w2nn.L1Criterion','nn.Criterion')

function L1Criterion:__init()
   parent.__init(self)
   self.diff = torch.Tensor()
   self.linear_loss_buff = torch.Tensor()
end
function L1Criterion:updateOutput(input, target)
   self.diff:resizeAs(input):copy(input)
   if input:dim() == 1 then
      self.diff[1] = input[1] - target
   else
      for i = 1, input:size(1) do
	 self.diff[i]:add(-1, target[i])
      end
   end
   local linear_targets = self.diff
   local linear_loss = self.linear_loss_buff:resizeAs(linear_targets):copy(linear_targets):abs():sum()
   self.output = (linear_loss) / input:nElement()
   return self.output
end
function L1Criterion:updateGradInput(input, target)
   local norm = 1.0 / input:nElement()
   self.gradInput:resizeAs(self.diff):copy(self.diff):sign():mul(norm)
   return self.gradInput
end
