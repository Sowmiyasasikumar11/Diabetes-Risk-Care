document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".flash").forEach((message) => {
        window.setTimeout(() => {
            message.style.opacity = "0";
            message.style.transition = "opacity .35s ease";
        }, 4500);
    });
});
