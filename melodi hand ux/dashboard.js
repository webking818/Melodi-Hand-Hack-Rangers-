const startBtn = document.getElementById("startPlayBtn");

if (startBtn) {
  startBtn.addEventListener("click", () => {
    document.body.classList.add("playing");
  });
}