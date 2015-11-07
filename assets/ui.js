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
	var style = $("input[name=style]:checked").val()
	if (style == "photo") {
	    $("input[name=noise]").prop("disabled", true);
	    $(".noise-field").hide()
	} else {
	    $("input[name=noise]").prop("disabled", false);
	    $(".noise-field").show();
	}
    }
    
    $("#url").change(clear_file);
    $("#file").change(clear_url);
    $("input[name=style]").change(on_change_style);
})
