require 'pl'
local pairwise_transform = {}

pairwise_transform = tablex.update(pairwise_transform, require('pairwise_transform_scale'))
pairwise_transform = tablex.update(pairwise_transform, require('pairwise_transform_jpeg'))
pairwise_transform = tablex.update(pairwise_transform, require('pairwise_transform_jpeg_scale'))
pairwise_transform = tablex.update(pairwise_transform, require('pairwise_transform_user'))

return pairwise_transform
