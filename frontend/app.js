const cameraGrid = document.getElementById("cameraGrid");
const alertsList = document.getElementById("alertsList");
const alertCount = document.getElementById("alertCount");
const connectionStatus = document.getElementById("connectionStatus");

const addCameraBtn = document.getElementById("addCameraBtn");
const addCameraModal = document.getElementById("addCameraModal");
const cancelAddCam = document.getElementById("cancelAddCam");
const confirmAddCam = document.getElementById("confirmAddCam");
const newCamName = document.getElementById("newCamName");
const newCamSource = document.getElementById("newCamSource");
const addCamError = document.getElementById("addCamError");

let totalAlerts = 0;

function tileId(name) {
  return `cam-${name.replace(/[^a-zA-Z0-9_-]/g, "_")}`;
}

function renderCameraTile(cam) {
  const id = tileId(cam.name);
  let tile = document.getElementById(id);
  if (!tile) {
    tile = document.createElement("div");
    tile.className = "camera-tile";
    tile.id = id;
    tile.innerHTML = `
      <img src="/video_feed/${encodeURIComponent(cam.name)}" alt="${cam.name}" />
      <div class="tile-footer">
        <span><span class="dot"></span><span class="cam-name">${cam.name}</span></span>
        <button class="remove-cam" data-name="${cam.name}">remove</button>
      </div>
    `;
    tile.querySelector(".remove-cam").addEventListener("click", () => removeCamera(cam.name));
    cameraGrid.appendChild(tile);
  }
  tile.classList.toggle("alert", !!cam.alert_active);
  const dot = tile.querySelector(".dot");
  dot.classList.toggle("connected", !!cam.connected);
  dot.classList.toggle("disconnected", !cam.connected);
}

async function refreshCameras() {
  try {
    const res = await fetch("/api/cameras");
    const cams = await res.json();
    const seen = new Set();
    cams.forEach((cam) => {
      renderCameraTile(cam);
      seen.add(tileId(cam.name));
    });
    // remove tiles for cameras that no longer exist
    [...cameraGrid.children].forEach((tile) => {
      if (!seen.has(tile.id)) tile.remove();
    });
    connectionStatus.textContent = `online · ${cams.length} camera${cams.length === 1 ? "" : "s"}`;
    connectionStatus.classList.add("online");
    connectionStatus.classList.remove("offline");
  } catch (e) {
    connectionStatus.textContent = "backend unreachable";
    connectionStatus.classList.add("offline");
    connectionStatus.classList.remove("online");
  }
}

function renderAlert(alert, prepend = true) {
  const card = document.createElement("div");
  card.className = "alert-card";
  const time = new Date(alert.timestamp).toLocaleTimeString();
  card.innerHTML = `
    <img src="/snapshots/${encodeURIComponent(alert.snapshot_file)}" alt="snapshot" />
    <div class="meta">
      <strong>${alert.camera}</strong>
      confidence ${(alert.confidence * 100).toFixed(0)}%<br/>
      ${time}
    </div>
  `;
  if (prepend) {
    alertsList.prepend(card);
  } else {
    alertsList.appendChild(card);
  }
  totalAlerts += 1;
  alertCount.textContent = totalAlerts;
}

async function loadAlertHistory() {
  try {
    const res = await fetch("/api/alerts?limit=50");
    const alerts = await res.json();
    // API returns newest-first; render oldest-first so the list ends up newest-first on screen
    [...alerts].reverse().forEach((a) => renderAlert(a, true));
  } catch (e) {
    console.error("Failed to load alert history", e);
  }
}

function connectWebSocket() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/ws/alerts`);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "alert") {
      renderAlert(msg.data, true);
      flashCameraTile(msg.data.camera);
    }
  };

  ws.onclose = () => {
    setTimeout(connectWebSocket, 2000); // auto-reconnect
  };

  // keep the connection alive through idle proxies
  setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send("ping");
  }, 20000);
}

function flashCameraTile(name) {
  const tile = document.getElementById(tileId(name));
  if (tile) tile.classList.add("alert");
}

async function removeCamera(name) {
  if (!confirm(`Remove camera "${name}"?`)) return;
  await fetch(`/api/cameras/${encodeURIComponent(name)}`, { method: "DELETE" });
  refreshCameras();
}

addCameraBtn.addEventListener("click", () => {
  newCamName.value = "";
  newCamSource.value = "";
  addCamError.textContent = "";
  addCameraModal.classList.remove("hidden");
});
cancelAddCam.addEventListener("click", () => addCameraModal.classList.add("hidden"));

confirmAddCam.addEventListener("click", async () => {
  const name = newCamName.value.trim();
  const source = newCamSource.value.trim();
  if (!name || !source) {
    addCamError.textContent = "Both fields are required.";
    return;
  }
  const res = await fetch("/api/cameras", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, source }),
  });
  if (res.ok) {
    addCameraModal.classList.add("hidden");
    refreshCameras();
  } else {
    const err = await res.json();
    addCamError.textContent = err.error || "Failed to add camera.";
  }
});

// initial load
refreshCameras();
loadAlertHistory();
connectWebSocket();
setInterval(refreshCameras, 4000);
