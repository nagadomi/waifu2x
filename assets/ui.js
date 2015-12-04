$(function (){
    var expires = 365;
    function clear_file() {
	var new_file = $("#file").clone();
	new_file.change(clear_url);
	$("#file").replaceWith(new_file);
    }
    function clear_url() {
	$("#url").val("")
    }
    function on_change_style(e) {
	$("input[name=style]").parents("label").each(
	    function (i, elm) {
		$(elm).css("font-weight", "normal");
	    });
	var checked = $("input[name=style]:checked");
	checked.parents("label").css("font-weight", "bold");
	if (checked.val() == "art") {
	    $("h1").text("waifu2x");
	} else {
	    $("h1").html("w<s>/a/</s>ifu2x");
	}
	$.cookie("style", checked.val(), {expires: expires});
    }
    function on_change_noise_level(e)
    {
	$("input[name=noise]").parents("label").each(
	    function (i, elm) {
		$(elm).css("font-weight", "normal");
	    });
	var checked = $("input[name=noise]:checked");
	if (checked.val() != 0) {
	    checked.parents("label").css("font-weight", "bold");
	}
	$.cookie("noise", checked.val(), {expires: expires});
    }
    function on_change_scale_factor(e)
    {
	$("input[name=scale]").parents("label").each(
	    function (i, elm) {
		$(elm).css("font-weight", "normal");
	    });
	var checked = $("input[name=scale]:checked");
	if (checked.val() != 0) {
	    checked.parents("label").css("font-weight", "bold");
	}
	$.cookie("scale", checked.val(), {expires: expires});
    }
    function restore_from_cookie()
    {
	if ($.cookie("style")) {
	    $("input[name=style]").filter("[value=" + $.cookie("style") + "]").prop("checked", true)
	}
	if ($.cookie("noise")) {
	    $("input[name=noise]").filter("[value=" + $.cookie("noise") + "]").prop("checked", true)
	}
	if ($.cookie("scale")) {
	    $("input[name=scale]").filter("[value=" + $.cookie("scale") + "]").prop("checked", true)
	}
    }
    
    $("#url").change(clear_file);
    $("#file").change(clear_url);
    $("input[name=style]").change(on_change_style);
    $("input[name=noise]").change(on_change_noise_level);
    $("input[name=scale]").change(on_change_scale_factor);

    restore_from_cookie();
    on_change_style();
    on_change_scale_factor();
    on_change_noise_level();
})
