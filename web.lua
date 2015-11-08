require 'pl'
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
local ROOT = path.dirname(__FILE__)
package.path = path.join(ROOT, "lib", "?.lua;") .. package.path
_G.TURBO_SSL = true

require 'w2nn'
local uuid = require 'uuid'
local ffi = require 'ffi'
local md5 = require 'md5'
local iproc = require 'iproc'
local reconstruct = require 'reconstruct'
local image_loader = require 'image_loader'

-- Notes:  turbo and xlua has different implementation of string:split().
--         Therefore, string:split() has conflict issue.
--         In this script, use turbo's string:split().
local turbo = require 'turbo'

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x-api")
cmd:text("Options:")
cmd:option("-port", 8812, 'listen port')
cmd:option("-gpu", 1, 'Device ID')
cmd:option("-thread", -1, 'number of CPU threads')
local opt = cmd:parse(arg)
cutorch.setDevice(opt.gpu)
torch.setdefaulttensortype('torch.FloatTensor')
if opt.thread > 0 then
   torch.setnumthreads(opt.thread)
end
if cudnn then
   cudnn.fastest = true
   cudnn.benchmark = false
end
local ART_MODEL_DIR = path.join(ROOT, "models", "anime_style_art_rgb")
local PHOTO_MODEL_DIR = path.join(ROOT, "models", "ukbench")
local art_noise1_model = torch.load(path.join(ART_MODEL_DIR, "noise1_model.t7"), "ascii")
local art_noise2_model = torch.load(path.join(ART_MODEL_DIR, "noise2_model.t7"), "ascii")
local art_scale2_model = torch.load(path.join(ART_MODEL_DIR, "scale2.0x_model.t7"), "ascii")
local photo_scale2_model = torch.load(path.join(PHOTO_MODEL_DIR, "scale2.0x_model.t7"), "ascii")

local CACHE_DIR = path.join(ROOT, "cache")
local MAX_NOISE_IMAGE = 2560 * 2560
local MAX_SCALE_IMAGE = 1280 * 1280
local CURL_OPTIONS = {
   request_timeout = 15,
   connect_timeout = 10,
   allow_redirects = true,
   max_redirects = 2
}
local CURL_MAX_SIZE = 2 * 1024 * 1024

local function valid_size(x, scale)
   if scale == 0 then
      return x:size(2) * x:size(3) <= MAX_NOISE_IMAGE
   else
      return x:size(2) * x:size(3) <= MAX_SCALE_IMAGE
   end
end

local function cache_url(url)
   local hash = md5.sumhexa(url)
   local cache_file = path.join(CACHE_DIR, "url_" .. hash)
   if path.exists(cache_file) then
      return image_loader.load_float(cache_file)
   else
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
	    local fp = io.open(cache_file, "wb")
	    local blob = res.body
	    fp:write(blob)
	    fp:close()
	    return image_loader.decode_float(blob)
	 end
      end
   end
   return nil, nil, nil
end
local function get_image(req)
   local file = req:get_argument("file", "")
   local url = req:get_argument("url", "")
   local blob = nil
   local img = nil
   local alpha = nil
   if file and file:len() > 0 then
      blob = file
      return image_loader.decode_float(blob)
   elseif url and url:len() > 0 then
      return cache_url(url)
   end
   return nil, nil, nil
end
local function convert(x, options)
   local cache_file = path.join(CACHE_DIR, options.prefix .. ".png")
   if path.exists(cache_file) then
      return image.load(cache_file)
   else
      if options.style == "art" then
	 if options.method == "scale" then
	    x = reconstruct.scale(art_scale2_model, 2.0, x)
	    w2nn.cleanup_model(art_scale2_model)
	 elseif options.method == "noise1" then
	    x = reconstruct.image(art_noise1_model, x)
	    w2nn.cleanup_model(art_noise1_model)
	 else -- options.method == "noise2"
	    x = reconstruct.image(art_noise2_model, x)
	    w2nn.cleanup_model(art_noise2_model)
	 end
      else -- photo
	 x = reconstruct.scale(photo_scale2_model, 2.0, x)
	 w2nn.cleanup_model(photo_scale2_model)
      end
      image.save(cache_file, x)
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
   local x, alpha, blob = get_image(self)
   local scale = tonumber(self:get_argument("scale", "0"))
   local noise = tonumber(self:get_argument("noise", "0"))
   local style = self:get_argument("style", "art")
   if style ~= "art" then
      style = "photo" -- style must be art or photo
   end
   if x and valid_size(x, scale) then
      if (noise ~= 0 or scale ~= 0) then
	 local hash = md5.sumhexa(blob)
	 if noise == 1 then
	    x = convert(x, {method = "noise1", style = style, prefix = style .. "_noise1_" .. hash})
	 elseif noise == 2 then
	    x = convert(x, {method = "noise2", style = style, prefix = style .. "_noise2_" .. hash})
	 end
	 if scale == 1 or scale == 2 then
	    if noise == 1 then
	       x = convert(x, {method = "scale", style = style, prefix = style .. "_noise1_scale_" .. hash})
	    elseif noise == 2 then
	       x = convert(x, {method = "scale", style = style, prefix = style .. "_noise2_scale_" .. hash})
	    else
	       x = convert(x, {method = "scale", style = style, prefix = style .. "_scale_" .. hash})
	    end
	    if scale == 1 then
	       x = iproc.scale(x,
			       math.floor(x:size(3) * (1.6 / 2.0) + 0.5),
			       math.floor(x:size(2) * (1.6 / 2.0) + 0.5),
			       "Jinc")
	    end
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
local index_ja = file.read(path.join(ROOT, "assets", "index.ja.html"))
local index_ru = file.read(path.join(ROOT, "assets", "index.ru.html"))
local index_en = file.read(path.join(ROOT, "assets", "index.html"))
function FormHandler:get()
   local lang = self.request.headers:get("Accept-Language")
   if lang then
      local langs = utils.split(lang, ",")
      for i = 1, #langs do
	 langs[i] = utils.split(langs[i], ";")[1]
      end
      if langs[1] == "ja" then
	 self:write(index_ja)
      elseif langs[1] == "ru" then
	 self:write(index_ru)
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
      {"^/style.css", turbo.web.StaticFileHandler, path.join(ROOT, "assets", "style.css")},
      {"^/ui.js", turbo.web.StaticFileHandler, path.join(ROOT, "assets", "ui.js")},
      {"^/index.html", turbo.web.StaticFileHandler, path.join(ROOT, "assets", "index.html")},
      {"^/index.ja.html", turbo.web.StaticFileHandler, path.join(ROOT, "assets", "index.ja.html")},
      {"^/index.ru.html", turbo.web.StaticFileHandler, path.join(ROOT, "assets", "index.ru.html")},
      {"^/api$", APIHandler},
   }
)
app:listen(opt.port, "0.0.0.0", {max_body_size = CURL_MAX_SIZE})
turbo.ioloop.instance():start()
