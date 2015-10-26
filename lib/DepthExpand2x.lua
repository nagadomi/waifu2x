if mynn.DepthExpand2x then
   return mynn.DepthExpand2x
end
local DepthExpand2x, parent = torch.class('mynn.DepthExpand2x','nn.Module')
 
function DepthExpand2x:__init()
   parent:__init()
end

function DepthExpand2x:updateOutput(input)
   local x = input
   -- (batch_size, depth, height, width)
   self.shape = x:size()

   assert(self.shape:size() == 4, "input must be 4d tensor")
   assert(self.shape[2] % 4 == 0, "depth must be depth % 4 = 0")
   -- (batch_size, width, height, depth)
   x = x:transpose(2, 4)
   -- (batch_size, width, height * 2, depth / 2)
   x = x:reshape(self.shape[1], self.shape[4], self.shape[3] * 2, self.shape[2] / 2)
   -- (batch_size, height * 2, width, depth / 2)
   x = x:transpose(2, 3)
   -- (batch_size, height * 2, width * 2, depth / 4)
   x = x:reshape(self.shape[1], self.shape[3] * 2, self.shape[4] * 2, self.shape[2] / 4)
   -- (batch_size, depth / 4, height * 2, width * 2)
   x = x:transpose(2, 4)
   x = x:transpose(3, 4)
   self.output:resizeAs(x):copy(x) -- contiguous
   
   return self.output
end

function DepthExpand2x:updateGradInput(input, gradOutput)
   -- (batch_size, depth / 4, height * 2, width * 2)
   local x = gradOutput
   -- (batch_size, height * 2, width * 2, depth / 4)
   x = x:transpose(2, 4)
   x = x:transpose(2, 3)
   -- (batch_size, height * 2, width, depth / 2)
   x = x:reshape(self.shape[1], self.shape[3] * 2, self.shape[4], self.shape[2] / 2)
   -- (batch_size, width, height * 2, depth / 2)
   x = x:transpose(2, 3)
   -- (batch_size, width, height, depth)
   x = x:reshape(self.shape[1], self.shape[4], self.shape[3], self.shape[2])
   -- (batch_size, depth, height, width)
   x = x:transpose(2, 4)
   
   self.gradInput:resizeAs(x):copy(x)
   
   return self.gradInput
end

function DepthExpand2x.test()
   require 'image'
   local function show(x)
      local img = torch.Tensor(3, x:size(3), x:size(4))
      img[1]:copy(x[1][1])
      img[2]:copy(x[1][2])
      img[3]:copy(x[1][3])
      image.display(img)
   end
   local img = image.lena()
   local x = torch.Tensor(1, img:size(1) * 4, img:size(2), img:size(3))
   for i = 0, img:size(1) * 4 - 1 do
      src_index = ((i % 3) + 1)
      x[1][i + 1]:copy(img[src_index])
   end
   show(x)
   
   local de2x = mynn.DepthExpand2x()
   out = de2x:forward(x)
   show(out)
   out = de2x:updateGradInput(x, out)
   show(out)
end
