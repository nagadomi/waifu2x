_G.TURBO_SSL = true
local turbo = require 'turbo'
local uuid = require 'uuid'
local ffi = require 'ffi'
local md5 = require 'md5'
require 'pl'
require './lib/portable'
require './lib/LeakyReLU'

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x-api")
cmd:text("Options:")
cmd:option("-port", 8812, 'listen port')
cmd:option("-gpu", 1, 'Device ID')
cmd:option("-core", 2, 'number of CPU cores')
local opt = cmd:parse(arg)
cutorch.setDevice(opt.gpu)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.core)

local iproc = require './lib/iproc'
local reconstruct = require './lib/reconstruct'
local image_loader = require './lib/image_loader'

local MODEL_DIR = "./models/anime_style_art_rgb"

local noise1_model = torch.load(path.join(MODEL_DIR, "noise1_model.t7"), "ascii")
local noise2_model = torch.load(path.join(MODEL_DIR, "noise2_model.t7"), "ascii")
local scale20_model = torch.load(path.join(MODEL_DIR, "scale2.0x_model.t7"), "ascii")

local USE_CACHE = true
local CACHE_DIR = "./cache"
local MAX_NOISE_IMAGE = 2560 * 2560
local MAX_SCALE_IMAGE = 1280 * 1280
local CURL_OPTIONS = {
   request_timeout = 15,
   connect_timeout = 10,
   allow_redirects = true,
   max_redirects = 2
}
local CURL_MAX_SIZE = 2 * 1024 * 1024
local BLOCK_OFFSET = 7 -- see srcnn.lua

local function valid_size(x, scale)
   if scale == 0 then
      return x:size(2) * x:size(3) <= MAX_NOISE_IMAGE
   else
      return x:size(2) * x:size(3) <= MAX_SCALE_IMAGE
   end
end

local function get_image(req)
   local file = req:get_argument("file", "")
   local url = req:get_argument("url", "")
   local blob = nil
   local img = nil
   local alpha = nil
   if file and file:len() > 0 then
      blob = file
      img, alpha = image_loader.decode_float(blob)
   elseif url and url:len() > 0 then
      local res = coroutine.yield(
	 turbo.async.HTTPClient({verify_ca=false},
				nil,
				CURL_MAX_SIZE):fetch(url, CURL_OPTIONS)
      )
      if res.code == 200 then
	 local content_type = res.headers:get("Content-Type", true)
	 if type(content_type) == "table" then
	    content_type = content_type[1]
	 end
	 if content_type and content_type:find("image") then
	    blob = res.body
	    img, alpha = image_loader.decode_float(blob)
	 end
      end
   end
   return img, blob, alpha
end

local function apply_denoise1(x)
   return reconstruct.image(noise1_model, x, BLOCK_OFFSET)
end
local function apply_denoise2(x)
   return reconstruct.image(noise2_model, x, BLOCK_OFFSET)
end
local function apply_scale2x(x)
   return reconstruct.scale(scale20_model, 2.0, x, BLOCK_OFFSET)
end
local function cache_do(cache, x, func)
   if path.exists(cache) then
      return image.load(cache)
   else
      x = func(x)
      image.save(cache, x)
      return x
   end
end

local function client_disconnected(handler)
   return not(handler.request and
		 handler.request.connection and
		 handler.request.connection.stream and
		 (not handler.request.connection.stream:closed()))
end

local APIHandler = class("APIHandler", turbo.web.RequestHandler)
function APIHandler:post()
   if client_disconnected(self) then
      self:set_status(400)
      self:write("client disconnected")
      return
   end
   local x, src, alpha = get_image(self)
   local scale = tonumber(self:get_argument("scale", "0"))
   local noise = tonumber(self:get_argument("noise", "0"))
   if x and valid_size(x, scale) then
      if USE_CACHE and (noise ~= 0 or scale ~= 0) then
	 local hash = md5.sumhexa(src)
	 local cache_noise1 = path.join(CACHE_DIR, hash .. "_noise1.png")
	 local cache_noise2 = path.join(CACHE_DIR, hash .. "_noise2.png")
	 local cache_scale = path.join(CACHE_DIR, hash .. "_scale.png")
	 local cache_noise1_scale = path.join(CACHE_DIR, hash .. "_noise1_scale.png")
	 local cache_noise2_scale = path.join(CACHE_DIR, hash .. "_noise2_scale.png")
	 
	 if noise == 1 then
	    x = cache_do(cache_noise1, x, apply_denoise1)
	 elseif noise == 2 then
	    x = cache_do(cache_noise2, x, apply_denoise2)
	 end
	 if scale == 1 or scale == 2 then
	    if noise == 1 then
	       x = cache_do(cache_noise1_scale, x, apply_scale2x)
	    elseif noise == 2 then
	       x = cache_do(cache_noise2_scale, x, apply_scale2x)
	    else
	       x = cache_do(cache_scale, x, apply_scale2x)
	    end
	    if scale == 1 then
	       x = iproc.scale(x,
			       math.floor(x:size(3) * (1.6 / 2.0) + 0.5),
			       math.floor(x:size(2) * (1.6 / 2.0) + 0.5),
			       "Jinc")
	    end
	 end
      elseif noise ~= 0 or scale ~= 0 then
	 if noise == 1 then
	    x = apply_denoise1(x)
	 elseif noise == 2 then
	    x = apply_denoise2(x)
	 end
	 if scale == 1 then
	    local x16 = {math.floor(x:size(3) * 1.6 + 0.5), math.floor(x:size(2) * 1.6 + 0.5)}
	    x = apply_scale2x(x)
	    x = iproc.scale(x, x16[1], x16[2], "Jinc")
	 elseif scale == 2 then
	    x = apply_scale2x(x)
	 end
      end
      local name = uuid() .. ".png"
      local blob, len = image_loader.encode_png(x, alpha)
      
      self:set_header("Content-Disposition", string.format('filename="%s"', name))
      self:set_header("Content-Type", "image/png")
      self:set_header("Content-Length", string.format("%d", len))
      self:write(ffi.string(blob, len))
   else
      if not x then
	 self:set_status(400)
	 self:write("ERROR: unsupported image format.")
      else
	 self:set_status(400)
	 self:write("ERROR: image size exceeds maximum allowable size.")
      end
   end
   collectgarbage()
end
local FormHandler = class("FormHandler", turbo.web.RequestHandler)
local index_ja = file.read("./assets/index.ja.html")
local index_en = file.read("./assets/index.html")
function FormHandler:get()
   local lang = self.request.headers:get("Accept-Language")
   if lang then
      local langs = utils.split(lang, ",")
      for i = 1, #langs do
	 langs[i] = utils.split(langs[i], ";")[1]
      end
      if langs[1] == "ja" then
	 self:write(index_ja)
      else
	 self:write(index_en)
      end
   else
      self:write(index_en)
   end
end
turbo.log.categories = {
   ["success"] = true,
   ["notice"] = false,
   ["warning"] = true,
   ["error"] = true,
   ["debug"] = false,
   ["development"] = false
}
local app = turbo.web.Application:new(
   {
      {"^/$", FormHandler},
      {"^/index.html", turbo.web.StaticFileHandler, path.join("./assets", "index.html")},
      {"^/index.ja.html", turbo.web.StaticFileHandler, path.join("./assets", "index.ja.html")},
      {"^/api$", APIHandler},
   }
)
app:listen(opt.port, "0.0.0.0", {max_body_size = CURL_MAX_SIZE})
turbo.ioloop.instance():start()
