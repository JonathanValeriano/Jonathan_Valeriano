document.addEventListener('DOMContentLoaded', () => {
    const carouselContainer = document.querySelector('.carousel-container');
    const slides = document.querySelectorAll('.slide');
    const totalSlides = slides.length / 2; // Considera apenas os slides originais
    const prevButton = document.querySelector('.prev');
    const nextButton = document.querySelector('.next');
    let index = 0;
    let autoplayInterval;
    let resetTimer;

    if (!carouselContainer || !prevButton || !nextButton || totalSlides === 0) {
        console.error("Elementos do carrossel não encontrados!");
        return;
    }

    // Atualiza a posição do carrossel
    function updateCarousel() {
        carouselContainer.style.transform = `translateX(-${index * 33.33}%)`;
    }

    // Navega para o próximo conjunto de slides
    function nextSlide() {
        index++;
        if (index >= totalSlides) {
            carouselContainer.style.transition = 'none';
            index = 0;
            updateCarousel();
            void carouselContainer.offsetWidth;
            carouselContainer.style.transition = 'transform 0.5s ease-in-out';
            index++;
        }
        updateCarousel();
        resetAutoplay(); // Reinicia o timer de inatividade
    }

    // Navega para o conjunto anterior de slides
    function prevSlide() {
        index--;
        if (index < 0) {
            carouselContainer.style.transition = 'none';
            index = totalSlides - 1;
            updateCarousel();
            void carouselContainer.offsetWidth;
            carouselContainer.style.transition = 'transform 0.5s ease-in-out';
            index--;
        }
        updateCarousel();
        resetAutoplay(); // Reinicia o timer de inatividade
    }

    // Reinicia o autoplay após 3 segundos de inatividade
    function resetAutoplay() {
        clearInterval(autoplayInterval);
        clearTimeout(resetTimer);
        resetTimer = setTimeout(() => {
            autoplayInterval = setInterval(nextSlide, 2000);
        }, 3000);
    }

    // Adiciona eventos aos botões
    prevButton.addEventListener('click', prevSlide);
    nextButton.addEventListener('click', nextSlide);

    // Inicia o autoplay
    autoplayInterval = setInterval(nextSlide, 2000);

    // Pausa o autoplay ao passar o mouse
    carouselContainer.addEventListener('mouseenter', () => {
        clearInterval(autoplayInterval);
        clearTimeout(resetTimer);
    });

    // Retoma o autoplay ao remover o mouse (se não houver interação recente)
    carouselContainer.addEventListener('mouseleave', () => {
        if (!resetTimer) {
            autoplayInterval = setInterval(nextSlide, 2000);
        }
    });
});