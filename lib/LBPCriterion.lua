-- Random Generated Local Binary Pattern Loss 
local LBPCriterion, parent = torch.class('w2nn.LBPCriterion','nn.Criterion')

local function create_filters(ch, n, k, layers)
   local model = nn.Sequential()
   for i = 1, layers do
      local n_input = ch
      if i > 1 then
	 n_input = n
      end
      local filter = w2nn.RandomBinaryConvolution(n_input, n, k, k)
      if i == 1 then
	 -- channel identity
	 for j = 1, ch do
	    filter.weight[j]:fill(0)
	    filter.weight[j][j][math.floor(k/2)+1][math.floor(k/2)+1] = 1
	 end
      end
      model:add(filter)
      --if layers > 1 and i ~= layers then
      --   model:add(nn.Sigmoid(true))
      --end
   end
   return model
end
function LBPCriterion:__init(ch, n, k, layers)
   parent.__init(self)
   self.layers = layers or 1
   self.gamma = 0.1
   self.n = n or 128
   self.k = k or 3
   self.ch = ch
   self.filter1 = create_filters(self.ch, self.n, self.k, self.layers)
   self.filter2 = self.filter1:clone()
   self.diff = torch.Tensor()
   self.diff_abs = torch.Tensor()
   self.square_loss_buff = torch.Tensor()
   self.linear_loss_buff = torch.Tensor()
   self.input = torch.Tensor()
   self.target = torch.Tensor()
end
function LBPCriterion:updateOutput(input, target)
   if input:dim() == 2 then
      local k = math.sqrt(input:size(2) / self.ch)
      input = input:reshape(input:size(1), self.ch, k, k)
   end
   if target:dim() == 2 then
      local k = math.sqrt(target:size(2) / self.ch)
      target = target:reshape(target:size(1), self.ch, k, k)
   end
   self.input:resizeAs(input):copy(input):clamp(0, 1)
   self.target:resizeAs(target):copy(target):clamp(0, 1)

   local lb1 = self.filter1:forward(self.input)
   local lb2 = self.filter2:forward(self.target)

   -- huber loss
   self.diff:resizeAs(lb1):copy(lb1)
   self.diff:add(-1, lb2)
   self.diff_abs:resizeAs(self.diff):copy(self.diff):abs()
   
   local square_targets = self.diff[torch.lt(self.diff_abs, self.gamma)]
   local linear_targets = self.diff[torch.ge(self.diff_abs, self.gamma)]
   local square_loss = self.square_loss_buff:resizeAs(square_targets):copy(square_targets):pow(2.0):mul(0.5):sum()
   local linear_loss = self.linear_loss_buff:resizeAs(linear_targets):copy(linear_targets):abs():add(-0.5 * self.gamma):mul(self.gamma):sum()
   --self.outlier_rate = linear_targets:nElement() / input:nElement()
   self.output = (square_loss + linear_loss) / lb1:nElement()

   return self.output
end

function LBPCriterion:updateGradInput(input, target)
   local d2 = false
   if input:dim() == 2 then
      d2 = true
      local k = math.sqrt(input:size(2) / self.ch)
      input = input:reshape(input:size(1), self.ch, k, k)
   end
   local norm = self.n / self.input:nElement()
   self.gradInput:resizeAs(self.diff):copy(self.diff):mul(norm)
   local outlier = torch.ge(self.diff_abs, self.gamma)
   self.gradInput[outlier] = torch.sign(self.diff[outlier]) * self.gamma * norm
   local grad_input = self.filter1:updateGradInput(input, self.gradInput)
   if d2 then
      grad_input = grad_input:reshape(grad_input:size(1), grad_input:size(2) * grad_input:size(3) * grad_input:size(4))
   end
   return grad_input
end
