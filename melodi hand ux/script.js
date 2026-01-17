const customCursor = document.querySelector('.custom-cursor');

window.addEventListener('mousemove', (e) => {
  customCursor.style.left = e.clientX + 'px';
  customCursor.style.top = e.clientY + 'px';
});
const startBtn = document.getElementById("startBtn");
if (startBtn) {
  startBtn.addEventListener("click", () => {
    window.location.href = "login.html";
  });
}
