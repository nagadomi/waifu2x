local ShakeShakeTable, parent = torch.class('w2nn.ShakeShakeTable','nn.Module')

function ShakeShakeTable:__init()
   parent.__init(self)
   self.alpha = torch.Tensor()
   self.beta = torch.Tensor()
   self.first = torch.Tensor()
   self.second = torch.Tensor()
   self.train = true
end
function ShakeShakeTable:updateOutput(input)
   local batch_size = input[1]:size(1)
   if self.train then
      self.alpha:resize(batch_size):uniform()
      self.beta:resize(batch_size):uniform()
      self.second:resizeAs(input[1]):copy(input[2])
      for i = 1, batch_size do
	 self.second[i]:mul(self.alpha[i])
      end
      self.output:resizeAs(input[1]):copy(input[1])
      for i = 1, batch_size do
	 self.output[i]:mul(1.0 - self.alpha[i])
      end
      self.output:add(self.second):mul(2)
   else
      self.output:resizeAs(input[1]):copy(input[1]):add(input[2])
   end
   return self.output
end
function ShakeShakeTable:updateGradInput(input, gradOutput)
   local batch_size = input[1]:size(1)
   self.first:resizeAs(gradOutput):copy(gradOutput)
   for i = 1, batch_size do
      self.first[i]:mul(self.beta[i])
   end
   self.second:resizeAs(gradOutput):copy(gradOutput)
   for i = 1, batch_size do
      self.second[i]:mul(1.0 - self.beta[i])
   end
   self.gradOutput = {self.first, self.second}
   return self.gradOutput
end
