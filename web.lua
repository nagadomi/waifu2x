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
local alpha_util = require 'alpha_util'
local compression = require 'compression'
local gm = require 'graphicsmagick'

-- Note:  turbo and xlua has different implementation of string:split().
--         Therefore, string:split() has conflict issue.
--         In this script, use turbo's string:split().
local turbo = require 'turbo'

local cmd = torch.CmdLine()
cmd:text()
cmd:text("waifu2x-api")
cmd:text("Options:")
cmd:option("-port", 8812, 'listen port')
cmd:option("-gpu", 1, 'Device ID')
cmd:option("-enable_tta", 0, 'enable TTA query(0|1)')
cmd:option("-crop_size", 256, 'patch size per process')
cmd:option("-batch_size", 1, 'batch size')
cmd:option("-thread", -1, 'number of CPU threads')
cmd:option("-force_cudnn", 0, 'use cuDNN backend (0|1)')
cmd:option("-max_pixels", 3000 * 3000, 'maximum number of output image pixels (e.g. 3000x3000=9000000)')
cmd:option("-curl_request_timeout", 60, "request_timeout for curl")
cmd:option("-curl_connect_timeout", 60, "connect_timeout for curl")
cmd:option("-curl_max_redirects", 2, "max_redirects for curl")
cmd:option("-max_body_size", 5 * 1024 * 1024, "maximum allowed size for uploaded files")
cmd:option("-cache_max", 200, "number of cached images on RAM")

local opt = cmd:parse(arg)
cutorch.setDevice(opt.gpu)
torch.setdefaulttensortype('torch.FloatTensor')
if opt.thread > 0 then
   torch.setnumthreads(opt.thread)
end
if cudnn then
   cudnn.fastest = true
   cudnn.benchmark = true
end
opt.force_cudnn = opt.force_cudnn == 1
opt.enable_tta = opt.enable_tta == 1
--local ART_MODEL_DIR = path.join(ROOT, "models", "upconv_7", "art")
local ART_MODEL_DIR = path.join(ROOT, "models", "cunet", "art")
local PHOTO_MODEL_DIR = path.join(ROOT, "models", "upconv_7", "photo")
local art_model = {
   scale = w2nn.load_model(path.join(ART_MODEL_DIR, "scale2.0x_model.t7"), opt.force_cudnn),
   noise0_scale = w2nn.load_model(path.join(ART_MODEL_DIR, "noise0_scale2.0x_model.t7"), opt.force_cudnn),
   noise1_scale = w2nn.load_model(path.join(ART_MODEL_DIR, "noise1_scale2.0x_model.t7"), opt.force_cudnn),
   noise2_scale = w2nn.load_model(path.join(ART_MODEL_DIR, "noise2_scale2.0x_model.t7"), opt.force_cudnn),
   noise3_scale = w2nn.load_model(path.join(ART_MODEL_DIR, "noise3_scale2.0x_model.t7"), opt.force_cudnn),
   noise0 = w2nn.load_model(path.join(ART_MODEL_DIR, "noise0_model.t7"), opt.force_cudnn),
   noise1 = w2nn.load_model(path.join(ART_MODEL_DIR, "noise1_model.t7"), opt.force_cudnn),
   noise2 = w2nn.load_model(path.join(ART_MODEL_DIR, "noise2_model.t7"), opt.force_cudnn),
   noise3 = w2nn.load_model(path.join(ART_MODEL_DIR, "noise3_model.t7"), opt.force_cudnn)
}
local photo_model = {
   scale = w2nn.load_model(path.join(PHOTO_MODEL_DIR, "scale2.0x_model.t7"), opt.force_cudnn),
   noise0_scale = w2nn.load_model(path.join(PHOTO_MODEL_DIR, "noise0_scale2.0x_model.t7"), opt.force_cudnn),
   noise1_scale = w2nn.load_model(path.join(PHOTO_MODEL_DIR, "noise1_scale2.0x_model.t7"), opt.force_cudnn),
   noise2_scale = w2nn.load_model(path.join(PHOTO_MODEL_DIR, "noise2_scale2.0x_model.t7"), opt.force_cudnn),
   noise3_scale = w2nn.load_model(path.join(PHOTO_MODEL_DIR, "noise3_scale2.0x_model.t7"), opt.force_cudnn),
   noise0 = w2nn.load_model(path.join(PHOTO_MODEL_DIR, "noise0_model.t7"), opt.force_cudnn),
   noise1 = w2nn.load_model(path.join(PHOTO_MODEL_DIR, "noise1_model.t7"), opt.force_cudnn),
   noise2 = w2nn.load_model(path.join(PHOTO_MODEL_DIR, "noise2_model.t7"), opt.force_cudnn),
   noise3 = w2nn.load_model(path.join(PHOTO_MODEL_DIR, "noise3_model.t7"), opt.force_cudnn)
}
collectgarbage()
local CLEANUP_MODEL = true -- if you are using the low memory GPU, you could use this flag.
local CACHE_DIR = path.join(ROOT, "cache")
local MAX_NOISE_IMAGE = opt.max_pixels
local MAX_SCALE_IMAGE = (math.sqrt(opt.max_pixels) / 2)^2
local PNG_DEPTH = 8
local CURL_OPTIONS = {
   request_timeout = opt.curl_request_timeout,
   connect_timeout = opt.curl_connect_timeout,
   allow_redirects = true,
   max_redirects = opt.curl_max_redirects
}
local CURL_MAX_SIZE = opt.max_body_size

local function valid_size(x, scale, tta_level)
   if scale <= 0 then
      local limit = math.pow(math.floor(math.pow(MAX_NOISE_IMAGE / tta_level, 0.5)), 2)
      return x:size(2) * x:size(3) <= limit
   else
      local limit = math.pow(math.floor(math.pow(MAX_SCALE_IMAGE / tta_level, 0.5)), 2)
      return x:size(2) * x:size(3) <= limit
   end
end
local function auto_tta_level(x, scale)
   local limit2, limit4, limit8
   if scale <= 0 then
      limit2 = math.pow(math.floor(math.pow(MAX_NOISE_IMAGE / 2, 0.5)), 2)
      limit4 = math.pow(math.floor(math.pow(MAX_NOISE_IMAGE / 4, 0.5)), 2)
      limit8 = math.pow(math.floor(math.pow(MAX_NOISE_IMAGE / 8, 0.5)), 2)
   else
      limit2 = math.pow(math.floor(math.pow(MAX_SCALE_IMAGE / 2, 0.5)), 2)
      limit4 = math.pow(math.floor(math.pow(MAX_SCALE_IMAGE / 4, 0.5)), 2)
      limit8 = math.pow(math.floor(math.pow(MAX_SCALE_IMAGE / 8, 0.5)), 2)
   end
   local px = x:size(2) * x:size(3)
   if px <= limit8 then
      return 8
   elseif px <= limit4 then
      return 4
   elseif px <= limit2 then
      return 2
   else
      return 1
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
   return nil, nil
end
local function get_image(req)
   local file_info = req:get_arguments("file")
   local url = req:get_argument("url", "")
   local file = nil
   local filename = nil
   if file_info and #file_info == 1 then
      file = file_info[1][1]
      local disp = file_info[1]["content-disposition"]
      if disp and disp["filename"] then
	 filename = path.basename(disp["filename"])
      end
   end
   if file and file:len() > 0 then
      local x, meta = image_loader.decode_float(file)
      return x, meta, filename
   elseif url and url:len() > 0 then
      local x, meta = cache_url(url)
      return x, meta, filename
   end
   return nil, nil, nil
end
local function cleanup_model(model)
   if CLEANUP_MODEL then
      model:clearState() -- release GPU memory
   end
end

-- cache
local g_cache = {}
local function cache_count()
   local count = 0
   for _ in pairs(g_cache) do
      count = count + 1
   end
   return count
end
local function cache_remove_old()
   local old_time = nil
   local old_key = nil
   for k, v in pairs(g_cache) do
      if old_time == nil or old_time > v.updated_at then
	 old_key = k
	 old_time = v.updated_at
      end
   end
   if old_key then
      g_cache[old_key] = nil
   end
end
local function cache_compress(raw_image)
   if raw_image then
      compressed_image = compression.compress(iproc.float2byte(raw_image))
      return compressed_image
   else
      return nil
   end
end
local function cache_decompress(compressed_image)
   if compressed_image then
      local raw_image = compression.decompress(compressed_image)
      return iproc.byte2float(raw_image)
   else
      return nil
   end
end
local function cache_get(filename)
   local cache = g_cache[filename]
   if cache then
      return {image = cache_decompress(cache.image),
	      alpha = cache_decompress(cache.alpha)}
   else
      return nil
   end
end
local function cache_put(filename, image, alpha)
   g_cache[filename] = {image = cache_compress(image),
			alpha = cache_compress(alpha),
			updated_at = os.time()};
   local count = cache_count(g_cache)
   if count > opt.cache_max then
      cache_remove_old()
   end
end
local function convert(x, meta, options)
   local cache_file = path.join(CACHE_DIR, options.prefix .. ".png")
   local alpha = meta.alpha
   local alpha_orig = alpha
   local cache = cache_get(cache_file)

   if cache then
      meta = tablex.copy(meta)
      meta.alpha = cache.alpha
      return cache.image, meta
   else
      local model = nil
      if options.style == "art" then
	 model = art_model
      elseif options.style == "photo" then
	 model = photo_model
      end
      if options.border then
	 x = alpha_util.make_border(x, alpha_orig, reconstruct.offset_size(model.scale))
      end
      if (options.method == "scale" or
	     options.method == "noise0_scale" or
	     options.method == "noise1_scale" or
	     options.method == "noise2_scale" or
	     options.method == "noise3_scale")
      then
	 x = reconstruct.scale_tta(model[options.method], options.tta_level, 2.0, x,
				   opt.crop_size, opt.batch_size)
	 if alpha then
	    if not (alpha:size(2) == x:size(2) and alpha:size(3) == x:size(3)) then
	       alpha = reconstruct.scale(model.scale, 2.0, alpha,
					 opt.crop_size, opt.batch_size)
	       cleanup_model(model.scale)
	    end
	 end
	 cleanup_model(model[options.method])
      elseif (options.method == "noise0" or
		 options.method == "noise1" or
		 options.method == "noise2" or
		 options.method == "noise3")
      then
	 x = reconstruct.image_tta(model[options.method], options.tta_level,
				   x, opt.crop_size, opt.batch_size)
	 cleanup_model(model[options.method])
      end
      cache_put(cache_file, x, alpha)
      meta = tablex.copy(meta)
      meta.alpha = alpha
      return x, meta
   end
end
local function client_disconnected(handler)
   return not(handler.request and
		 handler.request.connection and
		 handler.request.connection.stream and
		 (not handler.request.connection.stream:closed()))
end
local function make_output_filename(filename, mode)
   local e = path.extension(filename)
   local base = filename:sub(0, filename:len() - e:len())
   if mode then
      return base .. "_waifu2x_" .. mode .. ".png"
   else
      return base .. ".png"
   end
end

local APIHandler = class("APIHandler", turbo.web.RequestHandler)
function APIHandler:post()
   if client_disconnected(self) then
      self:set_status(400)
      self:write("client disconnected")
      return
   end
   local x, meta, filename = get_image(self)
   local scale = tonumber(self:get_argument("scale", "-1"))
   local noise = tonumber(self:get_argument("noise", "-1"))
   local tta_level = tonumber(self:get_argument("tta_level", "1"))
   local style = self:get_argument("style", "art")
   local download = (self:get_argument("download", "")):len()

   if client_disconnected(self) then
      self:set_status(400)
      self:write("client disconnected")
      return
   end
   if opt.enable_tta then
      if tta_level == 0 then
	 tta_level = auto_tta_level(x, scale)
      end
      if not (tta_level == 0 or tta_level == 1 or tta_level == 2 or tta_level == 4 or tta_level == 8) then
	 tta_level = 1
      end
   else
      tta_level = 1
   end
   if style ~= "art" then
      style = "photo" -- style must be art or photo
   end
   if x and valid_size(x, scale, tta_level) then
      local prefix = nil
      if (noise >= 0 or scale > 0) then
	 local hash = md5.sumhexa(meta.blob)
	 local alpha_prefix = style .. "_" .. hash .. "_alpha"
	 local border = false
	 if scale >= 0 and meta.alpha then
	    border = true
	 end
	 if (scale == 1 or scale == 2) and (noise < 0) then
	    prefix = style .. "_scale_tta_"  .. tta_level .. "_"
	    x, meta = convert(x, meta, {method = "scale",
					style = style,
					tta_level = tta_level,
					prefix = prefix .. hash,
					alpha_prefix = alpha_prefix,
					border = border})
	    if scale == 1 then
	       x = iproc.scale(x, x:size(3) * (1.6 / 2.0), x:size(2) * (1.6 / 2.0), "Sinc")
	    end
	 elseif (scale == 1 or scale == 2) and (noise == 0 or noise == 1 or noise == 2 or noise == 3) then
	    prefix = style .. string.format("_noise%d_scale_tta_", noise)  .. tta_level .. "_"
	    x, meta = convert(x, meta, {method = string.format("noise%d_scale", noise),
					style = style,
					tta_level = tta_level,
					prefix = prefix .. hash,
					alpha_prefix = alpha_prefix,
					border = border})
	    if scale == 1 then
	       x = iproc.scale(x, x:size(3) * (1.6 / 2.0), x:size(2) * (1.6 / 2.0), "Sinc")
	    end
	 elseif (noise == 0 or noise == 1 or noise == 2 or noise == 3) then
	    prefix = style .. string.format("_noise%d_tta_", noise) .. tta_level .. "_"
	    x = convert(x, meta, {method = string.format("noise%d", noise), 
				  style = style, 
				  tta_level = tta_level,
				  prefix = prefix .. hash,
				  alpha_prefix = alpha_prefix,
				  border = border})
	    border = false
	 end
      end
      local name = nil
      if filename then 
	 if prefix then
	    name = make_output_filename(filename, prefix:sub(0, prefix:len()-1))
	 else
	    name = make_output_filename(filename, nil)
	 end
      else
	 name = uuid() .. ".png"
      end
      local blob = image_loader.encode_png(alpha_util.composite(x, meta.alpha),
					   tablex.update({depth = PNG_DEPTH, inplace = true}, meta))

      self:set_header("Content-Length", string.format("%d", #blob))
      if download > 0 then
	 self:set_header("Content-Type", "application/octet-stream")
	 self:set_header("Content-Disposition", string.format('attachment; filename="%s"', name))
      else
	 self:set_header("Content-Type", "image/png")
	 self:set_header("Content-Disposition", string.format('inline; filename="%s"', name))
      end
      self:write(blob)
   else
      if not x then
	 self:set_status(400)
	 self:write("ERROR: An error occurred. (unsupported image format/connection timeout/file is too large)")
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
local index_pt = file.read(path.join(ROOT, "assets", "index.pt.html"))
local index_es = file.read(path.join(ROOT, "assets", "index.es.html"))
local index_fr = file.read(path.join(ROOT, "assets", "index.fr.html"))
local index_de = file.read(path.join(ROOT, "assets", "index.de.html"))
local index_tr = file.read(path.join(ROOT, "assets", "index.tr.html"))
local index_zh_cn = file.read(path.join(ROOT, "assets", "index.zh-CN.html"))
local index_zh_tw = file.read(path.join(ROOT, "assets", "index.zh-TW.html"))
local index_ko = file.read(path.join(ROOT, "assets", "index.ko.html"))
local index_nl = file.read(path.join(ROOT, "assets", "index.nl.html"))
local index_ca = file.read(path.join(ROOT, "assets", "index.ca.html"))
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
      elseif langs[1] == "pt" or langs[1] == "pt-BR" then
	 self:write(index_pt)
      elseif langs[1] == "es" or langs[1] == "es-ES" then
	 self:write(index_es)
      elseif langs[1] == "fr" then
	 self:write(index_fr)
      elseif langs[1] == "de" then
	 self:write(index_de)
      elseif langs[1] == "tr" then
	 self:write(index_tr)
      elseif langs[1] == "zh-CN" or langs[1] == "zh" then
	 self:write(index_zh_cn)
      elseif langs[1] == "zh-TW" then
	 self:write(index_zh_tw)
      elseif langs[1] == "ko" then
	 self:write(index_ko)
      elseif langs[1] == "nl" then
	 self:write(index_nl)
      elseif langs[1] == "ca" or langs[1] == "ca-ES" or langs[1] == "ca-FR" or langs[1] == "ca-IT" or langs[1] == "ca-AD" then
	 self:write(index_ca)
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
      {"^/api$", APIHandler},
      {"^/([%a%d%.%-_]+)$", turbo.web.StaticFileHandler, path.join(ROOT, "assets/")},
   }
)
app:listen(opt.port, "0.0.0.0", {max_body_size = CURL_MAX_SIZE})
turbo.ioloop.instance():start()
