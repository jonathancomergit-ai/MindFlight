// MindFlight viewer: draws N drones from the server's WebSocket stream.
//
// Each binary message is a Float32Array of N × 7 values, per drone: x, y, z, qw, qx, qy, qz
// (metres, world frame, Z up; quaternion rotates body → world). See mindflight/viewer/server.py.
//
// /config may also list drone groups (the live demo does). Each group gets its own back-rotor colour and a line in the
// legend. Nothing about groups goes over the WebSocket: the colours are worked out here from drone numbers.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

const VALUES_PER_DRONE = 7;

// Z up, like the simulator. Must be set before any camera or object is made.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

// ---------- scene ----------

const config = await (await fetch("/config")).json();

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x10141a);
scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 2.0));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(3, 2, 5);
scene.add(sun);

// Ground grid with 1 m squares. GridHelper lies in the x-z plane, so turn it flat onto x-y.
const gridSizeM = 2 * Math.ceil(Math.sqrt(config.n_drones) * 0.25 + 3);
const grid = new THREE.GridHelper(gridSizeM, gridSizeM, 0x5a6472, 0x2c333d);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
scene.add(new THREE.AxesHelper(1)); // x red, y green, z blue, 1 m long

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 500);
camera.position.set(-gridSizeM * 0.45, -gridSizeM * 0.35, gridSizeM * 0.3);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 1);
controls.update();

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------- one drone, at real scale ----------

// X layout from PLAN.md: motors at (±d, ±d) with d = arm / √2, so the arms run along the diagonals.
const armM = config.arm_m;
const armXyM = armM / Math.SQRT2;

function frameGeometry() {
  const arms = [Math.PI / 4, -Math.PI / 4].map((angle) => {
    const bar = new THREE.BoxGeometry(2 * armM, 0.005, 0.003);
    bar.rotateZ(angle);
    return bar;
  });
  const body = new THREE.BoxGeometry(0.03, 0.02, 0.008);
  return mergeGeometries([...arms, body]);
}

// Rotor discs for the front motors (m0, m3 at +x) or the back motors (m1, m2 at −x).
function rotorGeometry(front) {
  const positions = [[+armXyM, -armXyM], [-armXyM, -armXyM], [-armXyM, +armXyM], [+armXyM, +armXyM]]; // m0..m3
  const discs = positions
    .filter(([x]) => (x > 0) === front)
    .map(([x, y]) => {
      const disc = new THREE.CylinderGeometry(config.prop_radius_m, config.prop_radius_m, 0.001, 24);
      disc.rotateX(Math.PI / 2); // cylinder axis y -> z
      disc.translate(x, y, 0.004);
      return disc;
    });
  return mergeGeometries(discs);
}

// ---------- groups: each group colours its back rotors (the demo's 4 test groups) ----------

const GROUP_COLORS = [0x6fdc8c, 0xff8389, 0x4ea8ff, 0xc08cff]; // green, red, blue, purple
const PLAIN_ROTOR_COLOR = new THREE.Color(0xc8d0da); // light grey, for drones in no group
const groups = config.groups ?? []; // [{name, motors, expect, first, count}], drones first .. first + count - 1

function groupColor(groupIndex) {
  return new THREE.Color(GROUP_COLORS[groupIndex % GROUP_COLORS.length]);
}

function droneColor(droneIndex) {
  const groupIndex = groups.findIndex((g) => droneIndex >= g.first && droneIndex < g.first + g.count);
  return groupIndex >= 0 ? groupColor(groupIndex) : PLAIN_ROTOR_COLOR;
}

// ---------- the swarm: one InstancedMesh per part, so thousands of drones are three draw calls ----------

// Front rotors are always orange so you can see which way each drone faces. The back rotors' material is white,
// so each drone's own instance colour (its group colour, or light grey) shows unchanged.
const parts = [
  { geometry: frameGeometry(), material: new THREE.MeshStandardMaterial({ color: 0x3b4450, roughness: 0.6 }) },
  { geometry: rotorGeometry(true), material: new THREE.MeshStandardMaterial({ color: 0xff9f43, roughness: 0.5 }) },
  { geometry: rotorGeometry(false), material: new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 }) },
];
const BACK_ROTORS = 2; // index in `parts` of the part coloured per drone

let capacity = 0;
let meshes = []; // one InstancedMesh per part

function ensureCapacity(nDrones) {
  if (nDrones <= capacity) return;
  for (const mesh of meshes) {
    scene.remove(mesh);
    mesh.dispose();
  }
  capacity = Math.max(nDrones, 2 * capacity);
  meshes = parts.map(({ geometry, material }) => {
    const mesh = new THREE.InstancedMesh(geometry, material, capacity);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false; // drones move, so the default bounding sphere goes stale
    mesh.count = 0;
    scene.add(mesh);
    return mesh;
  });
  for (let i = 0; i < capacity; i++) meshes[BACK_ROTORS].setColorAt(i, droneColor(i));
}

const sizeToggle = document.getElementById("size-toggle");
const position = new THREE.Vector3();
const rotation = new THREE.Quaternion();
const scale = new THREE.Vector3();
const matrix = new THREE.Matrix4();

function showFrame(poses) {
  const nDrones = Math.floor(poses.length / VALUES_PER_DRONE);
  ensureCapacity(nDrones);
  const s = sizeToggle.checked ? 10 : 1;
  scale.set(s, s, s);
  for (let i = 0; i < nDrones; i++) {
    const o = i * VALUES_PER_DRONE;
    position.set(poses[o], poses[o + 1], poses[o + 2]);
    rotation.set(poses[o + 4], poses[o + 5], poses[o + 6], poses[o + 3]); // Three.js wants (x, y, z, w)
    matrix.compose(position, rotation, scale);
    for (const mesh of meshes) mesh.setMatrixAt(i, matrix);
  }
  for (const mesh of meshes) {
    mesh.count = nDrones;
    mesh.instanceMatrix.needsUpdate = true;
  }
  hud.drones.textContent = nDrones;
}

// ---------- HUD ----------

const hud = {
  drones: document.getElementById("hud-drones"),
  renderFps: document.getElementById("hud-render-fps"),
  streamFps: document.getElementById("hud-stream-fps"),
  status: document.getElementById("hud-status"),
};

function setStatus(text) {
  hud.status.textContent = text;
  hud.status.className = `status-${text}`;
}

// Legend: one line per group, "■ name  motors → what you should see".
const legend = document.getElementById("legend");
groups.forEach((group, groupIndex) => {
  const row = document.createElement("div");
  row.className = "legend-row";
  const swatch = document.createElement("span");
  swatch.className = "legend-swatch";
  swatch.style.background = `#${groupColor(groupIndex).getHexString()}`;
  const name = document.createElement("b");
  name.textContent = group.name;
  const motors = document.createElement("span");
  motors.textContent = group.motors;
  const expect = document.createElement("span");
  expect.className = "legend-expect";
  expect.textContent = `→ ${group.expect}`;
  row.append(swatch, name, motors, expect);
  legend.append(row);
});
legend.hidden = groups.length === 0;

// Restart: the button or the R key asks the server to start the sim over (POST /reset).
async function restartSim() {
  await fetch("/reset", { method: "POST" });
}
if (config.resettable) {
  document.getElementById("reset-row").hidden = false;
  document.getElementById("reset-button").addEventListener("click", restartSim);
  window.addEventListener("keydown", (event) => {
    const plainKey = !event.ctrlKey && !event.metaKey && !event.altKey;
    if (plainKey && (event.key === "r" || event.key === "R")) restartSim();
  });
  if (config.restart_after_s) {
    document.getElementById("loop-note").textContent = `starts over every ${config.restart_after_s} s`;
  }
}

// ---------- WebSocket stream ----------

let latestPoses = null; // newest frame not drawn yet
let framesReceived = 0;

function connect() {
  setStatus("connecting");
  const socket = new WebSocket(`ws://${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  socket.onopen = () => setStatus("connected");
  socket.onmessage = (event) => {
    latestPoses = new Float32Array(event.data);
    framesReceived++;
  };
  socket.onclose = () => {
    setStatus("disconnected");
    setTimeout(connect, 1000); // keep trying, so restarting the server just works
  };
}
connect();

// ---------- render loop ----------

let framesRendered = 0;
let lastHudUpdateMs = performance.now();
let lastPoses = null;

sizeToggle.addEventListener("change", () => { if (lastPoses) showFrame(lastPoses); });

renderer.setAnimationLoop((nowMs) => {
  if (latestPoses) {
    showFrame(latestPoses);
    lastPoses = latestPoses;
    latestPoses = null;
  }
  controls.update();
  renderer.render(scene, camera);
  framesRendered++;

  const elapsedMs = nowMs - lastHudUpdateMs;
  if (elapsedMs >= 500) {
    hud.renderFps.textContent = Math.round((framesRendered * 1000) / elapsedMs);
    hud.streamFps.textContent = Math.round((framesReceived * 1000) / elapsedMs);
    framesRendered = framesReceived = 0;
    lastHudUpdateMs = nowMs;
  }
});
