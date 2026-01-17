/* ==============================
   PAGE CONFIG (DEFENSIVE)
================================ */

const pageType = document.body.dataset.page;

// SAFE DEFAULTS (never undefined)
let density = 12000;
let mouseRadius = 100;

if (pageType === "hero") {
  density = 2000;
  mouseRadius = 180;
} else if (pageType === "login") {
  density = 3000;
  mouseRadius = 70;
} else if (pageType === "dashboard") {
  density = 3000;
  mouseRadius = 120;
}

/* ==============================
   CANVAS SETUP
================================ */

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

window.addEventListener("resize", () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  init(); // re-init particles on resize
});

/* ==============================
   MOUSE TRACKING (SAFE INIT)
================================ */

const mouse = {
  x: canvas.width / 2,
  y: canvas.height / 2,
  radius: mouseRadius
};

window.addEventListener("mousemove", (e) => {
  mouse.x = e.clientX;
  mouse.y = e.clientY;
});

/* ==============================
   PARTICLE CLASS
================================ */

class Particle {
  constructor(x, y, size) {
    this.x = x;
    this.y = y;
    this.size = size;
    this.baseX = x;
    this.baseY = y;
    this.density = Math.random() * 30 + 1;
  }

  draw() {
    ctx.fillStyle = "rgba(180, 220, 255, 0.8)";
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.closePath();
    ctx.fill();
  }

  update() {
    const dx = mouse.x - this.x;
    const dy = mouse.y - this.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance < mouse.radius) {
      const force = (mouse.radius - distance) / mouse.radius;
      const directionX = dx / distance;
      const directionY = dy / distance;

      this.x -= directionX * force * this.density;
      this.y -= directionY * force * this.density;
    } else {
      // smooth return
      this.x += (this.baseX - this.x) / 10;
      this.y += (this.baseY - this.y) / 10;
    }
  }
}

/* ==============================
   PARTICLE INIT
================================ */

const particles = [];

function init() {
  particles.length = 0;

  const numberOfParticles = Math.min(
    (canvas.width * canvas.height) / density,
    400
  );

  for (let i = 0; i < numberOfParticles; i++) {
    const x = Math.random() * canvas.width;
    const y = Math.random() * canvas.height;
    const size = Math.random() * 1.5 + 0.5; // denser look, cheaper render
    particles.push(new Particle(x, y, size));
  }
}

init();

/* ==============================
   ANIMATION LOOP
================================ */

function animate() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  particles.forEach((p) => {
    p.update();
    p.draw();
  });

  requestAnimationFrame(animate);
}

animate();