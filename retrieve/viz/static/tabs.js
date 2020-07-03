
function removeActive() {
  $("li").each(function() {
    $(this).removeClass("is-active");
  });
}

function hideAll(){
  $("#heatmap-content").addClass("is-hidden");
  $("#doc-content").addClass("is-hidden");
}

function switchToTab(tab) {
  removeActive();
  hideAll();
  $("#" + tab + "-tab").addClass("is-active");
  $("#" + tab + "-content").removeClass("is-hidden");
}
