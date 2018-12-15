require 'cunn'
local EdgeFilter, parent = torch.class('w2nn.EdgeFilter', 'nn.SpatialConvolution')

function EdgeFilter:__init(nInputPlane)
   local output = 0
   parent.__init(self, nInputPlane, nInputPlane * 8, 3, 3, 1, 1, 0, 0)
end
function EdgeFilter:reset()
   self.bias = nil
   self.gradBias = nil	
   self.gradWeight:fill(0)
   self.weight:fill(0)
   local fi = 1

   -- each channel
   for ch = 1, self.nInputPlane do
      for i = 0, 8 do
	 y = math.floor(i / 3) + 1
	 x = i % 3 + 1
	 if not (y == 2 and x == 2) then
	    self.weight[fi][ch][2][2] = 1
	    self.weight[fi][ch][y][x] = -1
	    fi = fi + 1
	 end
      end
   end
end
function EdgeFilter:accGradParameters(input, gradOutput, scale)
end
function EdgeFilter:updateParameters(learningRate)
end
