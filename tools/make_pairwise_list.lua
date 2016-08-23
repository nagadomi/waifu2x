require 'pl'
local cjson = require 'cjson'

local function pairwise_from_entries(y_dir, x_dirs)
   local list = {}
   local y_files = dir.getfiles(y_dir, "*")
   for i, y_file in ipairs(y_files) do
      local basename = path.basename(y_file)
      local x_files = {}
      for i = 1, #x_dirs do
	 local x_file = path.join(x_dirs[i], basename)
	 if path.exists(x_file) then
	    table.insert(x_files, x_file)
	 end
      end
      if #x_files == 1 then
	 table.insert(list, {y =  y_file, x = x_files[1]})
      elseif #x_files > 1 then
	 local r = torch.random(1, #x_files)
	 table.insert(list, {y =  y_file, x = x_files[r]})
      end
   end
   return list
end
local function pairwise_from_list(y_dir, x_dirs, basename_file)
   local list = {}
   local basenames = utils.split(file.read(basename_file), "\n")
   for i, basename in ipairs(basenames) do
      local basename = path.basename(basename)
      local y_file = path.join(y_dir, basename)
      if path.exists(y_file) then
	 local x_files = {}
	 for i = 1, #x_dirs do
	    local x_file = path.join(x_dirs[i], basename)
	    if path.exists(x_file) then
	       table.insert(x_files, x_file)
	    end
	 end
	 if #x_files == 1 then
	    table.insert(list, {y =  y_file, x = x_files[1]})
	 elseif #x_files > 1 then
	    local r = torch.random(1, #x_files)
	    table.insert(list, {y =  y_file, x = x_files[r]})
	 end
      end
   end
   return list
end
local function output(list, filters, rate)
   local n = math.floor(#list * rate)
   if #list > 0 and n == 0 then
      n = 1
   end
   local perm = torch.randperm(#list)
   if #filters == 0 then
      filters = nil
   end
   for i = 1, n do
      local v = list[perm[i]]
      io.stdout:write('"' .. v.y:gsub('"', '""') .. '"' .. "," .. '"' .. cjson.encode({x = v.x, filters = filters}):gsub('"', '""') .. '"' .. "\n")
   end
end
local function get_xdirs(opt)
   local x_dirs = {}
   for k,v in pairs(opt) do
      local s, e = k:find("x_dir")
      if s == 1 then
	 table.insert(x_dirs, v)
      end
   end
   return x_dirs
end

local cmd = torch.CmdLine()
cmd:text("waifu2x make_pairwise_list")
cmd:option("-x_dir", "", 'Specify the directory for x(input)')
cmd:option("-y_dir", "", 'Specify the directory for y(groundtruth). The filenames should be same as x_dir')
cmd:option("-rate", 1, 'sampling rate')
cmd:option("-file_list", "", 'Specify the basename list (optional)')
cmd:option("-filters", "", 'Specify the downsampling filters')
cmd:option("-x_dir1", "", 'x for random choice')
cmd:option("-x_dir2", "", 'x for random choice')
cmd:option("-x_dir3", "", 'x for random choice')
cmd:option("-x_dir4", "", 'x for random choice')
cmd:option("-x_dir5", "", 'x for random choice')
cmd:option("-x_dir6", "", 'x for random choice')
cmd:option("-x_dir7", "", 'x for random choice')
cmd:option("-x_dir8", "", 'x for random choice')
cmd:option("-x_dir9", "", 'x for random choice')

torch.manualSeed(71)

local opt = cmd:parse(arg)

local x_dirs = get_xdirs(opt)

if opt.y_dir:len() == 0 or #x_dirs == 0 then
   cmd:help()
   os.exit(1)
end

local list
if opt.file_list:len() > 0 then
   list = pairwise_from_list(opt.y_dir, x_dirs, opt.file_list)
else
   list = pairwise_from_entries(opt.y_dir, x_dirs)
end
output(list, utils.split(opt.filters, ","), opt.rate)
