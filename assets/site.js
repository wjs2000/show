
(function () {
  function runCarousel(root, slideSelector, dotSelector, interval) {
    var slides = Array.prototype.slice.call(root.querySelectorAll(slideSelector));
    var dots = Array.prototype.slice.call(root.querySelectorAll(dotSelector));
    if (slides.length < 2) return;
    var index = 0;
    function show(next) {
      slides[index].classList.remove("is-active");
      if (dots[index]) dots[index].classList.remove("is-active");
      index = (next + slides.length) % slides.length;
      slides[index].classList.add("is-active");
      if (dots[index]) dots[index].classList.add("is-active");
    }
    dots.forEach(function (dot, i) {
      dot.addEventListener("click", function () { show(i); });
    });
    window.setInterval(function () { show(index + 1); }, interval);
  }
  document.querySelectorAll("[data-hero-carousel]").forEach(function (root) {
    runCarousel(root, ".hero-slide", ".hero-dots button", 4200);
  });
})();
