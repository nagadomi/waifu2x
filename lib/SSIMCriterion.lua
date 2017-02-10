-- SSIM Index, ref: http://www.cns.nyu.edu/~lcv/ssim/ssim_index.m
local SSIMCriterion, parent = torch.class('w2nn.SSIMCriterion','nn.Criterion')
function SSIMCriterion:__init(ch, kernel_size, sigma)
   parent.__init(self)
   local function gaussian2d(kernel_size, sigma)
      sigma = sigma or 1
      local kernel = torch.Tensor(kernel_size, kernel_size)
      local u = math.floor(kernel_size / 2) + 1
      local amp = (1 / math.sqrt(2 * math.pi * sigma^2))
      for x = 1, kernel_size do
	 for y = 1, kernel_size do
	 kernel[x][y] = amp * math.exp(-((x - u)^2 + (y - u)^2) / (2 * sigma^2))
	 end
      end
      kernel:div(kernel:sum())
      return kernel
   end
   ch = ch or 1
   kernel_size = kernel_size or 11
   sigma = sigma or 1.5
   local kernel = gaussian2d(kernel_size, sigma)
   if ch > 1 then
      local kernel_nd = torch.Tensor(ch, ch, kernel_size, kernel_size)
      for i = 1, ch do
	 for j = 1, ch do
	    kernel_nd[i][j]:copy(kernel)
	    if i ~= j then
	       kernel_nd[i][j]:zero()
	    end
	 end
      end
      kernel = kernel_nd
   end
   self.c1 = 0.01^2
   self.c2 = 0.03^2
   self.ch = ch
   self.conv = nn.SpatialConvolution(ch, ch, kernel_size, kernel_size, 1, 1, 0, 0):noBias()
   self.conv.weight:copy(kernel)
   self.mu1 = torch.Tensor()
   self.mu2 = torch.Tensor()
   self.mu1_sq = torch.Tensor()
   self.mu2_sq = torch.Tensor()
   self.mu1_mu2 = torch.Tensor()
   self.sigma1_sq = torch.Tensor()
   self.sigma2_sq = torch.Tensor()
   self.sigma12 = torch.Tensor()
   self.ssim_map = torch.Tensor()
end
function SSIMCriterion:updateOutput(input, target)-- dynamic range: 0-1
   assert(input:nElement() == target:nElement())
   local valid = self.conv:forward(input)
   self.mu1:resizeAs(valid):copy(valid)
   self.mu2:resizeAs(valid):copy(self.conv:forward(target))
   self.mu1_sq:resizeAs(self.mu1):copy(self.mu1):cmul(self.mu1)
   self.mu2_sq:resizeAs(self.mu2):copy(self.mu2):cmul(self.mu2)
   self.mu1_mu2:resizeAs(self.mu1):copy(self.mu1):cmul(self.mu2)
   self.sigma1_sq:resizeAs(valid):copy(self.conv:forward(torch.cmul(input, input)):add(-1, self.mu1_sq))
   self.sigma2_sq:resizeAs(valid):copy(self.conv:forward(torch.cmul(target, target)):add(-1, self.mu2_sq))
   self.sigma12:resizeAs(valid):copy(self.conv:forward(torch.cmul(input, target)):add(-1, self.mu1_mu2))

   local ssim = self.mu1_mu2:mul(2):add(self.c1):cmul(self.sigma12:mul(2):add(self.c2)):
      cdiv(self.mu1_sq:add(self.mu2_sq):add(self.c1):cmul(self.sigma1_sq:add(self.sigma2_sq):add(self.c2))):mean()
   return ssim
end
function SSIMCriterion:updateGradInput(input, target)
   error("not implemented")
end
