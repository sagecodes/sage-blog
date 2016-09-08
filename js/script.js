var subject = document.location.hash;
var subjectclass = '.' + subject.slice(1);

// Closes mobile drop down menu when nav link clicked
$(function () {
    $('.navbar-collapse ul li a:not(.dropdown-toggle)').click(function () {
        $('.navbar-toggle:visible').click();
    });
});


// Shows and hides portfolio projects
$(document).ready(function(){
    $(".all").addClass('show');


    // Show posts based on anchor in URL
    if(document.location.hash) {
        $(".all").removeClass('show');
        $(subjectclass +'-tag').addClass('show');
        $(subjectclass +'-btn').addClass('active');
    } else {
        $(".all").addClass('show');
    }

});






