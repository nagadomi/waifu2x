require 'pl'

CACHE_DIR="cache"
TTL = 3600 * 24

local files = dir.getfiles(CACHE_DIR, "*.png")
local now = os.time()
for i, f in pairs(files) do
   if now - path.getmtime(f) > TTL then
      file.delete(f)
   end
end
