$(function (){
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
    }
    function on_change_white_noise(e)
    {
	$("input[name=white_noise]").parents("label").each(
	    function (i, elm) {
		$(elm).css("font-weight", "normal");
	    });
	var checked = $("input[name=white_noise]:checked");
	if (checked.val() != 0) {
	    checked.parents("label").css("font-weight", "bold");
	}
    }
    function on_click_experimental_button(e)
    {
	if ($(this).hasClass("close")) {
	    $(".experimental .container").show();
	    $(this).removeClass("close");
	} else {
	    $(".experimental .container").hide();
	    $(this).addClass("close");
	}
	e.preventDefault();
	e.stopPropagation();
    }
    
    $("#url").change(clear_file);
    $("#file").change(clear_url);
    //$("input[name=style]").change(on_change_style);
    $("input[name=noise]").change(on_change_noise_level);
    $("input[name=scale]").change(on_change_scale_factor);
    //$("input[name=white_noise]").change(on_change_white_noise);
    //$(".experimental .button").click(on_click_experimental_button)
    
    //on_change_style();
    on_change_scale_factor();
    on_change_noise_level();
})
