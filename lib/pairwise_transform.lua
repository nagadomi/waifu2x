require 'pl'
local pairwise_transform = {}

pairwise_transform = tablex.update(pairwise_transform, require('pairwise_transform_scale'))
pairwise_transform = tablex.update(pairwise_transform, require('pairwise_transform_jpeg'))

print(pairwise_transform)

return pairwise_transform
