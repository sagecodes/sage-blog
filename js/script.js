var subject = document.location.hash;
var subjectclass = '.' + subject.slice(1);

// Closes mobile drop down menu when nav link clicked
$(function () {
            $('.navbar-collapse ul li a:not(.dropdown-toggle)').click(function () {
                    $('.navbar-toggle:visible').click();
            });
    });


// Shows and hides portfolio projects
// I know I know, its quick and "dirty, but it works for now"
$(document).ready(function(){
    $(".all").addClass('show');
    $(".all-btn").click(function(){
        $(".all").removeClass('show');
        $(".all").addClass('show');
    });
    $(".html-btn").click(function(){
        $(".all").removeClass('show');
        $(".html-tag").addClass('show');
    });
    $(".css-btn").click(function(){
        $(".all").removeClass('show');
        $(".css-tag").addClass('show');
    });
    $(".python-btn").click(function(){
        $(".all").removeClass('show');
        $(".python-tag").addClass('show');
    });
    $(".javascript-btn").click(function(){
        $(".all").removeClass('show');
        $(".javascript-tag").addClass('show');
    });
    $(".ruby-btn").click(function(){
        $(".all").removeClass('show');
        $(".ruby-tag").addClass('show');
    });
    $(".flask-btn").click(function(){
        $(".all").removeClass('show');
        $(".flask-tag").addClass('show');
    });
    $(".sql-btn").click(function(){
        $(".all").removeClass('show');
        $(".sql-tag").addClass('show');
    });
    $(".aws-btn").click(function(){
        $(".all").removeClass('show');
        $(".aws-tag").addClass('show');
    });


    // WIP Of replacing above and below toggle with efficient solution:

    // $(".post_tag").click(function(){
    // subject = document.location.hash;
    // subjectclass = '.' + subject.slice(1);

    //   if(document.location.hash) {
    //     $(".all").removeClass('show');
    //     $(subjectclass +'-tag').addClass('show');
    // }

    // });

    // Show posts with class clicked on from index
    if(document.location.hash) {
    $(".all").removeClass('show');
    $(subjectclass +'-tag').addClass('show');
}

});






