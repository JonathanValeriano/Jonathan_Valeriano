document.addEventListener('DOMContentLoaded', () => {
    initializeCarousel('.carousel-container', '.prev', '.next', 2000, true);
});

function initializeCarousel(containerSelector, prevButtonSelector, nextButtonSelector, autoplayDelay, hasIndicators = false) {
    const carouselContainer = document.querySelector(containerSelector);
    const slides = document.querySelectorAll(`${containerSelector} .slide`);
    const prevButton = document.querySelector(prevButtonSelector);
    const nextButton = document.querySelector(nextButtonSelector);
    const indicators = hasIndicators ? document.querySelectorAll(`${containerSelector} .carousel-indicators .indicator`) : null;

    if (!carouselContainer || !prevButton || !nextButton || slides.length === 0) {
        console.error("Elementos do carrossel não encontrados!");
        return;
    }

    const totalSlides = slides.length / 2; // Considera apenas os slides originais
    let index = 0;
    let autoplayInterval;

    // Atualiza a posição do carrossel e os indicadores (se houver)
    function updateCarousel() {
        const slideWidth = 100 / 3; // 3 slides visíveis por vez
        carouselContainer.style.transform = `translateX(-${index * slideWidth}%)`;

        // Atualiza os indicadores (se houver)
        if (hasIndicators && indicators) {
            indicators.forEach((indicator, i) => {
                indicator.classList.toggle('active', i === index);
            });
        }
    }

    // Navega para o próximo slide
    function nextSlide() {
        index = (index + 1) % totalSlides;
        if (index === 0) {
            // Reinicia a transição ao voltar ao primeiro slide
            carouselContainer.style.transition = 'none';
            updateCarousel();
            void carouselContainer.offsetWidth; // Força o reflow
            carouselContainer.style.transition = 'transform 0.5s ease-in-out';
        }
        updateCarousel();
    }

    // Navega para o slide anterior
    function prevSlide() {
        index = (index - 1 + totalSlides) % totalSlides;
        if (index === totalSlides - 1) {
            // Reinicia a transição ao ir para o último slide
            carouselContainer.style.transition = 'none';
            updateCarousel();
            void carouselContainer.offsetWidth; // Força o reflow
            carouselContainer.style.transition = 'transform 0.5s ease-in-out';
        }
        updateCarousel();
    }

    // Navega para um slide específico
    function goToSlide(slideIndex) {
        index = slideIndex;
        updateCarousel();
    }

    // Inicia o autoplay
    function startAutoplay() {
        autoplayInterval = setInterval(nextSlide, autoplayDelay);
    }

    // Pausa o autoplay
    function pauseAutoplay() {
        clearInterval(autoplayInterval);
    }

    // Reinicia o autoplay após um tempo de inatividade
    function resetAutoplay() {
        pauseAutoplay();
        setTimeout(startAutoplay, autoplayDelay);
    }

    // Adiciona eventos aos botões
    prevButton.addEventListener('click', () => {
        prevSlide();
        resetAutoplay();
    });

    nextButton.addEventListener('click', () => {
        nextSlide();
        resetAutoplay();
    });

    // Adiciona eventos aos indicadores (se houver)
    if (hasIndicators && indicators) {
        indicators.forEach((indicator, i) => {
            indicator.addEventListener('click', () => {
                goToSlide(i);
                resetAutoplay();
            });
        });
    }

    // Pausa o autoplay ao passar o mouse
    carouselContainer.addEventListener('mouseenter', pauseAutoplay);
    carouselContainer.addEventListener('mouseleave', startAutoplay);

    // Inicia o autoplay pela primeira vez
    startAutoplay();
}