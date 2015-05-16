if nn.LeakyReLU then
   return
end
local LeakyReLU, parent = torch.class('nn.LeakyReLU','nn.Module')
 
function LeakyReLU:__init(negative_scale)
   parent.__init(self)
   self.negative_scale = negative_scale or 0.333
   self.negative = torch.Tensor()
end
 
function LeakyReLU:updateOutput(input)
   self.output:resizeAs(input):copy(input):abs():add(input):div(2)
   self.negative:resizeAs(input):copy(input):abs():add(-1.0, input):mul(-0.5*self.negative_scale)
   self.output:add(self.negative)
   
   return self.output
end
 
function LeakyReLU:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput)
   -- filter positive
   self.negative:sign():add(1)
   torch.cmul(self.gradInput, gradOutput, self.negative)
   -- filter negative
   self.negative:add(-1):mul(-1 * self.negative_scale):cmul(gradOutput)
   self.gradInput:add(self.negative)
   
   return self.gradInput
end
