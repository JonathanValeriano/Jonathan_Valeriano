document.addEventListener('DOMContentLoaded', () => {
    const carousels = [
        {
            container: document.querySelector('.carousel-container'),
            slides: document.querySelectorAll('.slide'),
            prevButton: document.querySelector('.prev'),
            nextButton: document.querySelector('.next'),
            totalSlides: document.querySelectorAll('.slide').length / 2,
        },
        {
            container: document.querySelector('#dashboards .carousel-container'),
            slides: document.querySelectorAll('#dashboards .slide'),
            prevButton: document.querySelector('#dashboards .prev'),
            nextButton: document.querySelector('#dashboards .next'),
            indicators: document.querySelectorAll('#dashboards .carousel-indicators .indicator'),
            totalSlides: document.querySelectorAll('#dashboards .slide').length / 2,
        }
    ];

    carousels.forEach(carousel => initCarousel(carousel));

    function initCarousel({ container, slides, prevButton, nextButton, indicators, totalSlides }) {
        if (!container || !prevButton || !nextButton || totalSlides === 0) {
            console.error("Elementos do carrossel não encontrados!");
            return;
        }

        let index = 0;
        let autoplayInterval;
        let resetTimer;

        const updateCarousel = () => {
            const slideWidth = 100 / (slides.length / totalSlides);
            container.style.transform = `translateX(-${index * slideWidth}%)`;
            indicators?.forEach((indicator, i) => {
                indicator.classList.toggle('active', i === index);
            });
        };

        const nextSlide = () => {
            index = (index + 1) % totalSlides;
            updateCarousel();
        };

        const prevSlide = () => {
            index = (index - 1 + totalSlides) % totalSlides;
            updateCarousel();
        };

        const goToSlide = (slideIndex) => {
            index = slideIndex;
            updateCarousel();
        };

        const startAutoplay = (interval = 5000) => {
            autoplayInterval = setInterval(nextSlide, interval);
        };

        const pauseAutoplay = () => {
            clearInterval(autoplayInterval);
        };

        const resetAutoplay = (delay = 2000) => {
            pauseAutoplay();
            clearTimeout(resetTimer);
            resetTimer = setTimeout(() => startAutoplay(), delay);
        };

        prevButton.addEventListener('click', () => {
            prevSlide();
            resetAutoplay();
        });

        nextButton.addEventListener('click', () => {
            nextSlide();
            resetAutoplay();
        });

        indicators?.forEach((indicator, i) => {
            indicator.addEventListener('click', () => goToSlide(i));
        });

        container.addEventListener('mouseenter', pauseAutoplay);
        container.addEventListener('mouseleave', resetAutoplay);

        startAutoplay(totalSlides === 3 ? 5000 : 2000);
    }
});
