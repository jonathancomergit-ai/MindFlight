// MindFlight viewer: draws N drones from the server's WebSocket stream.
//
// Each binary message is a Float32Array of N × 7 values, per drone: x, y, z, qw, qx, qy, qz
// (metres, world frame, Z up; quaternion rotates body → world). See mindflight/viewer/server.py.
//
// The server may also send JSON text messages: the task's objects (targets, gates, pillars), info lines, learning
// charts and a leaderboard. See "Text messages" in server.py.
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

// ---------- the task's objects: targets, gates, pillars (from "scene" text messages) ----------

const sceneGroup = new THREE.Group();
scene.add(sceneGroup);
const GATE_COLORS = { next: new THREE.Color(0x6fdc8c), later: new THREE.Color(0xff9f43), done: new THREE.Color(0x4a5260) };
const GATE_BAR_M = 0.04; // drawn frame thickness
const PILLAR_HEIGHT_M = 4.0;

function clearScene() {
  for (const child of [...sceneGroup.children]) {
    sceneGroup.remove(child);
    child.geometry.dispose();
    child.material.dispose();
  }
}

function colored(geometry, color) {
  const colors = new Float32Array(geometry.attributes.position.count * 3);
  for (let i = 0; i < colors.length; i += 3) color.toArray(colors, i);
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return geometry;
}

// One square gate: 4 bars around an opening of `size`, standing upright, facing along (cos yaw, sin yaw, 0).
function gateGeometry(gate) {
  const half = gate.size / 2 + GATE_BAR_M / 2;
  const span = gate.size + 2 * GATE_BAR_M;
  const bars = [
    [new THREE.BoxGeometry(GATE_BAR_M, span, GATE_BAR_M), 0, +half], // top
    [new THREE.BoxGeometry(GATE_BAR_M, span, GATE_BAR_M), 0, -half], // bottom
    [new THREE.BoxGeometry(GATE_BAR_M, GATE_BAR_M, span), +half, 0], // left
    [new THREE.BoxGeometry(GATE_BAR_M, GATE_BAR_M, span), -half, 0], // right
  ].map(([bar, y, z]) => {
    bar.translate(0, y, z);
    bar.rotateZ(gate.yaw);
    bar.translate(gate.c[0], gate.c[1], gate.c[2]);
    return colored(bar, GATE_COLORS[gate.state] ?? GATE_COLORS.later);
  });
  return mergeGeometries(bars);
}

function showScene(message) {
  clearScene();
  const points = [];
  if (message.targets?.length) {
    const sphere = new THREE.SphereGeometry(0.03, 12, 8);
    const material = new THREE.MeshBasicMaterial({ color: 0xf1c21b });
    const mesh = new THREE.InstancedMesh(sphere, material, message.targets.length);
    const place = new THREE.Matrix4();
    message.targets.forEach(([x, y, z], i) => {
      mesh.setMatrixAt(i, place.makeTranslation(x, y, z));
      points.push([x, y]);
    });
    sceneGroup.add(mesh);
  }
  if (message.gates?.length) {
    const geometry = mergeGeometries(message.gates.map(gateGeometry));
    sceneGroup.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6 })));
    for (const gate of message.gates) points.push([gate.c[0], gate.c[1]]);
  }
  if (message.pillars?.length) {
    const geometry = mergeGeometries(
      message.pillars.map(([x, y, radius]) => {
        const pillar = new THREE.CylinderGeometry(radius, radius, PILLAR_HEIGHT_M, 20);
        pillar.rotateX(Math.PI / 2); // cylinder axis y -> z
        pillar.translate(x, y, PILLAR_HEIGHT_M / 2);
        return pillar;
      }),
    );
    sceneGroup.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0x8a94a3, roughness: 0.8 })));
    for (const [x, y] of message.pillars) points.push([x, y]);
  }
  fitView(points);
}

// The first time we learn how big the world is, grow the floor grid and point the camera at the middle of it.
let viewFitted = false;
function fitView(points) {
  if (viewFitted || points.length === 0) return;
  viewFitted = true;
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const centreX = (minX + maxX) / 2;
  const centreY = (minY + maxY) / 2;
  const extent = Math.max(maxX - minX, maxY - minY, 4); // metres the camera has to take in
  const reach = Math.max(Math.abs(minX), Math.abs(maxX), Math.abs(minY), Math.abs(maxY));
  const size = 2 * Math.ceil(reach + 2);
  if (size > gridSizeM) grid.scale.setScalar(size / gridSizeM);
  controls.target.set(centreX, centreY, 1);
  camera.position.set(centreX - extent * 0.55, centreY - extent * 0.45, extent * 0.45 + 1);
  camera.far = Math.max(camera.far, size * 4);
  camera.updateProjectionMatrix();
  controls.update();
}

// ---------- info lines, charts and the leaderboard (text messages) ----------

const infoBox = document.getElementById("info");
function showInfo(lines) {
  infoBox.replaceChildren(...lines.map((line) => Object.assign(document.createElement("div"), { textContent: line })));
  infoBox.hidden = lines.length === 0;
}

const chartBox = document.getElementById("charts");
let chartSeries = []; // [{key, label, canvas, labelEl}]
const chartPoints = []; // every metrics point so far
function showChartSeries(series) {
  chartSeries = series.map((s) => ({ ...s }));
  chartBox.replaceChildren(
    ...chartSeries.map((s) => {
      const panel = document.createElement("div");
      panel.className = "chart";
      s.labelEl = Object.assign(document.createElement("div"), { className: "chart-label", textContent: s.label });
      s.canvas = Object.assign(document.createElement("canvas"), { width: 260, height: 64 });
      panel.append(s.labelEl, s.canvas);
      return panel;
    }),
  );
  chartBox.hidden = chartSeries.length === 0;
  drawCharts();
}

function drawCharts() {
  for (const s of chartSeries) {
    const values = chartPoints.filter((p) => p[s.key] !== undefined).map((p) => [p.iteration, p[s.key]]);
    const ctx = s.canvas.getContext("2d");
    const { width, height } = s.canvas;
    ctx.clearRect(0, 0, width, height);
    if (values.length === 0) continue;
    const xs = values.map((v) => v[0]);
    const ys = values.map((v) => v[1]);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs, xMin + 1);
    let yMin = Math.min(...ys);
    let yMax = Math.max(...ys);
    if (yMax - yMin < 1e-9) [yMin, yMax] = [yMin - 1, yMax + 1];
    const px = (x) => 4 + ((x - xMin) / (xMax - xMin)) * (width - 8);
    const py = (y) => height - 4 - ((y - yMin) / (yMax - yMin)) * (height - 8);
    ctx.strokeStyle = "#4ea8ff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    values.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(px(x), py(y)) : ctx.lineTo(px(x), py(y))));
    ctx.stroke();
    const last = ys[ys.length - 1];
    const shown = Math.abs(last) < 10 ? last.toFixed(3) : last.toFixed(1);
    s.labelEl.textContent = `${s.label}: ${shown}  (iteration ${xs[xs.length - 1]})`;
  }
}

const boardBox = document.getElementById("board");
function showBoard(message) {
  const title = Object.assign(document.createElement("div"), { className: "board-title", textContent: message.title });
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const column of message.columns) head.append(Object.assign(document.createElement("th"), { textContent: column }));
  const body = table.createTBody();
  for (const row of message.rows) {
    const tr = body.insertRow();
    row.forEach((cell, i) => {
      const td = tr.insertCell();
      if (i === 0 && typeof cell === "string" && cell.startsWith("#")) {
        // The first cell may be "#rrggbb|name": a colour swatch, then the name.
        const [color, name] = cell.split("|");
        const swatch = Object.assign(document.createElement("span"), { className: "legend-swatch" });
        swatch.style.background = color;
        td.append(swatch, document.createTextNode(` ${name}`));
      } else {
        td.textContent = cell;
      }
    });
  }
  boardBox.replaceChildren(title, table);
  boardBox.hidden = false;
}

function handleText(message) {
  if (message.type === "scene") showScene(message);
  else if (message.type === "info") showInfo(message.lines);
  else if (message.type === "charts") showChartSeries(message.series);
  else if (message.type === "metrics") {
    chartPoints.push(...message.points);
    drawCharts();
  } else if (message.type === "board") showBoard(message);
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
    if (typeof event.data === "string") {
      handleText(JSON.parse(event.data)); // scene, info, charts, metrics, board: see server.py
      return;
    }
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
