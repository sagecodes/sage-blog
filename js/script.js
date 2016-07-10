// Closes mobile drop down menu when nav link clicked
$(function () {
            $('.navbar-collapse ul li a:not(.dropdown-toggle)').click(function () {
                    $('.navbar-toggle:visible').click();
            });
    });
