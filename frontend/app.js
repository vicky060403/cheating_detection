const cameraGrid = document.getElementById("cameraGrid");
const alertsList = document.getElementById("alertsList");
const alertCount = document.getElementById("alertCount");
const connectionStatus = document.getElementById("connectionStatus");

const addCameraBtn = document.getElementById("addCameraBtn");
const addCameraModal = document.getElementById("addCameraModal");
const cancelAddCam = document.getElementById("cancelAddCam");
const testCameraBtn = document.getElementById("testCameraBtn");
const confirmAddCam = document.getElementById("confirmAddCam");
const newCamName = document.getElementById("newCamName");
const newCamSource = document.getElementById("newCamSource");
const addCamError = document.getElementById("addCamError");
const snapshotModal = document.getElementById("snapshotModal");
const snapshotPreview = document.getElementById("snapshotPreview");
const closeSnapshot = document.getElementById("closeSnapshot");
const verifySnapshot = document.getElementById("verifySnapshot");
const snapshotStatus = document.getElementById("snapshotStatus");

let totalAlerts = 0;
let openSnapshotFile = null;

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
  card.className = `alert-card${alert.verified ? " verified" : ""}`;
  card.dataset.snapshotFile = alert.snapshot_file;
  const time = new Date(alert.timestamp).toLocaleTimeString();
  card.innerHTML = `
    <img class="snapshot-thumb" src="/snapshots/${encodeURIComponent(alert.snapshot_file)}" alt="Open alert snapshot for ${alert.camera}" tabindex="0" />
    <div class="meta">
      <strong>${alert.camera}</strong>
      confidence ${(alert.confidence * 100).toFixed(0)}%<br/>
      ${time}
      <span class="verification-status${alert.verified ? " verified" : ""}">${alert.verified ? "Verified" : "Awaiting review"}</span>
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

function openSnapshot(image) {
  openSnapshotFile = image.closest(".alert-card").dataset.snapshotFile;
  snapshotPreview.src = image.src;
  snapshotPreview.alt = image.alt;
  const card = image.closest(".alert-card");
  const isVerified = card.classList.contains("verified");
  snapshotStatus.textContent = isVerified ? "This snapshot has been verified." : "Teacher review required.";
  verifySnapshot.disabled = isVerified;
  verifySnapshot.textContent = isVerified ? "Verified" : "Mark as verified";
  snapshotModal.classList.remove("hidden");
  closeSnapshot.focus();
}

function hideSnapshot() {
  snapshotModal.classList.add("hidden");
  snapshotPreview.removeAttribute("src");
  openSnapshotFile = null;
}

function updateVerificationState(snapshotFile) {
  const card = [...alertsList.querySelectorAll(".alert-card")]
    .find((item) => item.dataset.snapshotFile === snapshotFile);
  if (!card) return;
  card.classList.add("verified");
  const status = card.querySelector(".verification-status");
  status.textContent = "Verified";
  status.classList.add("verified");
}

async function markSnapshotVerified() {
  if (!openSnapshotFile) return;
  const response = await fetch(`/api/alerts/${encodeURIComponent(openSnapshotFile)}/verify`, { method: "POST" });
  if (!response.ok) {
    snapshotStatus.textContent = "Could not save verification.";
    return;
  }
  updateVerificationState(openSnapshotFile);
  snapshotStatus.textContent = "This snapshot has been verified.";
  verifySnapshot.disabled = true;
  verifySnapshot.textContent = "Verified";
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
  addCamError.classList.remove("success");
  addCameraModal.classList.remove("hidden");
});
cancelAddCam.addEventListener("click", () => addCameraModal.classList.add("hidden"));

alertsList.addEventListener("click", (event) => {
  const image = event.target.closest(".snapshot-thumb");
  if (image) openSnapshot(image);
});
alertsList.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    const image = event.target.closest(".snapshot-thumb");
    if (image) {
      event.preventDefault();
      openSnapshot(image);
    }
  }
});
closeSnapshot.addEventListener("click", hideSnapshot);
verifySnapshot.addEventListener("click", markSnapshotVerified);
snapshotModal.addEventListener("click", (event) => {
  if (event.target === snapshotModal) hideSnapshot();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !snapshotModal.classList.contains("hidden")) hideSnapshot();
});

confirmAddCam.addEventListener("click", async () => {
  const name = newCamName.value.trim();
  const source = newCamSource.value.trim();
  if (!name || !source) {
    addCamError.textContent = "Both fields are required.";
    addCamError.classList.remove("success");
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

testCameraBtn.addEventListener("click", async () => {
  const source = newCamSource.value.trim();
  if (!source) {
    addCamError.textContent = "Enter an RTSP URL or camera source first.";
    return;
  }
  testCameraBtn.disabled = true;
  testCameraBtn.textContent = "Testing...";
  addCamError.textContent = "";
  try {
    const res = await fetch("/api/cameras/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    });
    const result = await res.json();
    addCamError.textContent = res.ok ? "Connection successful." : (result.error || "Connection failed.");
    addCamError.classList.toggle("success", res.ok);
  } catch (error) {
    addCamError.textContent = "Backend unreachable.";
    addCamError.classList.remove("success");
  } finally {
    testCameraBtn.disabled = false;
    testCameraBtn.textContent = "Test connection";
  }
});

// initial load
refreshCameras();
loadAlertHistory();
connectWebSocket();
setInterval(refreshCameras, 4000);
