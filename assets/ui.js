$(function (){
    var g_expires = 365;
    var g_max_noise_image = 2560 * 2560;
    var g_max_scale_image = 1280 * 1280;

    function clear_file() {
	var new_file = $("#file").clone();
	new_file.change(clear_url);
	$("#file").replaceWith(new_file);
    }
    function clear_url() {
	$("#url").val("")
    }
    function on_change_style(e) {
	var checked = $("input[name=style]:checked");
	if (checked.val() == "art") {
	    $(".main-title").text("waifu2x");
	} else {
	    $(".main-title").html("w<s>/a/</s>ifu2x");
	}
	$.cookie("style", checked.val(), {expires: g_expires});
    }
    function on_click_tta_rule(e) {
	e.preventDefault();
	e.stopPropagation();
	$(".tta_rule_text").toggle();
    }
    function on_change_tta_level(e) {
	var checked = $("input[name=tta_level]:checked");
	$.cookie("tta_level", checked.val(), {expires: g_expires});
	var level = checked.val();
	if (level == 0) {
	    level = 1;
	}
	var max_noise_w = Math.floor(Math.pow(g_max_noise_image / level, 0.5));
	var max_scale_w = Math.floor(Math.pow(g_max_scale_image / level, 0.5));
	var limit_text = $(".file_limits").text();
	var hits = 0;
	limit_text = limit_text.replace(/\d+x\d+/g, function() {
	    hits += 1;
	    if (hits == 1) {
		return "" + max_noise_w + "x" + max_noise_w;
	    } else {
		return "" + max_scale_w + "x" + max_scale_w;
	    }
	});
	$(".file_limits").text(limit_text);
	if (level == 1) {
	    $(".file_limits").css("color", "");
	} else {
	    $(".file_limits").css("color", "blue");
	}
    }
    function on_change_noise_level(e)
    {
	var checked = $("input[name=noise]:checked");
	$.cookie("noise", checked.val(), {expires: g_expires});
    }
    function on_change_scale_factor(e)
    {
	var checked = $("input[name=scale]:checked");
	$.cookie("scale", checked.val(), {expires: g_expires});
    }
    function restore_from_cookie()
    {
	if ($.cookie("style")) {
	    $("input[name=style]").filter("[value=" + $.cookie("style") + "]").prop("checked", true);
	}
	if ($.cookie("noise")) {
	    $("input[name=noise]").filter("[value=" + $.cookie("noise") + "]").prop("checked", true);
	}
	if ($.cookie("scale")) {
	    $("input[name=scale]").filter("[value=" + $.cookie("scale") + "]").prop("checked", true);
	}
	if ($.cookie("tta_level")) {
	    $("input[name=tta_level]").filter("[value=" + $.cookie("tta_level") + "]").prop("checked", true);
	}
    }
    function uuid() 
    {
	// ref: http://stackoverflow.com/a/2117523
	return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
	    var r = Math.random()*16|0, v = c == 'x' ? r : (r&0x3|0x8);
	    return v.toString(16);
	});
    }
    function download_with_xhr(e) 
    {
	if (typeof window.URL.createObjectURL == "undefined" ||
	    typeof window.Blob == "undefined" ||
	    typeof window.XMLHttpRequest == "undefined" ||
	    typeof window.URL.revokeObjectURL == "undefined")
	{
	    return;
	}
	$("input[name=download]").attr("disabled", "disabled");
	e.preventDefault();
	e.stopPropagation();
	var xhr = new XMLHttpRequest();
	xhr.open('POST', '/api', true);
	xhr.responseType = 'arraybuffer';
	xhr.onload= function(e) {
	    if (this.status == 200) {
		var blob = new Blob([this.response], {type : 'image/png'});
		var a = document.createElement("a");
		var url = URL.createObjectURL(blob);
		a.href = url;
		a.target = "_blank";
		a.download = uuid() + ".png";
		a.click();
		URL.revokeObjectURL(url);
		$("input[name=download]").removeAttr("disabled");
	    } else {
		alert("Download Error");
		$("input[name=download]").removeAttr("disabled");
	    }
	};
	xhr.send(new FormData($("form").get(0)));
    }
    function set_param()
    {
	var uri = URI(window.location.href);
	var url = uri.query(true)["url"];
	var style = uri.query(true)["style"];
	var noise = uri.query(true)["noise"];
	var scale = uri.query(true)["scale"];
	if (url) {
	    $("input[name=url]").val(url);
	}
	if (style) {
	    $("input[name=style]").filter("[value=" + style + "]").prop("checked", true);
	}
	if (noise) {
	    $("input[name=noise]").filter("[value=" + noise + "]").prop("checked", true);
	}
	if (scale) {
	    $("input[name=scale]").filter("[value=" + scale + "]").prop("checked", true);
	}
    }

    $("#url").change(clear_file);
    $("#file").change(clear_url);
    $("input[name=style]").change(on_change_style);
    $("input[name=noise]").change(on_change_noise_level);
    $("input[name=scale]").change(on_change_scale_factor);
    $("input[name=tta_level]").change(on_change_tta_level);
    $("input[name=download]").click(download_with_xhr);
    $("a.tta_rule").click(on_click_tta_rule);

    restore_from_cookie();
    on_change_style();
    on_change_scale_factor();
    on_change_noise_level();
    if ($("input[name=tta_level]").length > 0) {
	on_change_tta_level();
    }
    set_param();
})
