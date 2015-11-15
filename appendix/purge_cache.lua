require 'pl'

CACHE_DIR="cache"
TTL = 3600 * 24

local files = {}
local image_cache = dir.getfiles(CACHE_DIR, "*.png")
local url_cache = dir.getfiles(CACHE_DIR, "url_*")
for i = 1, #image_cache do 
   table.insert(files, image_cache[i])
end
for i = 1, #url_cache do 
   table.insert(files, url_cache[i])
end
local now = os.time()
for i, f in pairs(files) do
   if now - path.getmtime(f) > TTL then
      file.delete(f)
   end
end
