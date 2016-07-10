// add background to navbar on scroll
jQuery(window).scroll(function(){
    var fromTopPx = 55; // distance to trigger
    var scrolledFromtop = jQuery(window).scrollTop();
    if(scrolledFromtop > fromTopPx){
        jQuery('.navbar').addClass('scrolled');
    }else{
        jQuery('.navbar').removeClass('scrolled');
    }
});
