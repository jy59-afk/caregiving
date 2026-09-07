/* ==========================================================================
   app.js — the whole client. No framework.

   Flow: user types -> POST /api/chat with the full transcript -> render the
   reply bubble, the ranked photo cards, the map pins, and (on the
   matches_ready outcome) the copy-only enquiry draft. Every turn re-runs the
   pipeline server-side, so the cards + map just re-render from the new
   response — that's the "auto-refresh as preferences get clearer".
   ========================================================================== */

"use strict";

// ---- element handles -----------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const transcriptEl = $("#transcript");
const chatScrollEl = $("#chatScroll");
const composerEl = $("#composer");
const inputEl = $("#input");
const sendBtn = $("#send");
const cardGridEl = $("#cardGrid");
const cardsTitleEl = $("#cardsTitle");
const cardsCountEl = $("#cardsCount");
const cardsNoteEl = $("#cardsNote");
const draftMountEl = $("#draftMount");
const mapEl = $("#map");
const mapEmptyEl = $("#mapEmpty");

// ---- client state ------------------------------------------------------
const state = {
  messages: [],      // [{role, content}] — the full transcript, sent every turn
  matches: [],       // last rendered MatchOut[]
  map: null,         // Leaflet map instance (created lazily)
  tileLayer: null,
  markers: [],       // Leaflet markers, index-aligned with state.matches
  activeRank: null,  // which card/pin is highlighted
  busy: false,
  totalServices: 0,  // dataset size, for the "4 of N" featured caption
  showingFeatured: false,  // true until the first chat turn replaces the sample
};

// Roughly the extent of Singapore — the initial map view, before any matches.
const SG_BOUNDS = [[1.205, 103.59], [1.475, 104.06]];

// Emoji fallback per care type, used when a photo fails to load.
const TYPE_EMOJI = {
  "day-care": "🌤️",
  "home-help": "🏠",
  "short-stay": "🛏️",
  "night-respite": "🌙",
  "weekend-respite": "📅",
  respite: "🧡",
};

// ==========================================================================
// Rendering — transcript
// ==========================================================================

const scrollChatToBottom = () => { chatScrollEl.scrollTop = chatScrollEl.scrollHeight; };

function renderTranscript() {
  transcriptEl.replaceChildren();
  for (const m of state.messages) {
    const b = document.createElement("div");
    b.className = `bubble ${m.role}`;
    b.textContent = m.content;
    transcriptEl.appendChild(b);
  }
  scrollChatToBottom();
}

function showPending() {
  const b = document.createElement("div");
  b.className = "bubble assistant pending";
  b.id = "pendingBubble";
  b.textContent = "Matching options…";
  transcriptEl.appendChild(b);
  scrollChatToBottom();
}

function clearPending() {
  document.getElementById("pendingBubble")?.remove();
}

// ==========================================================================
// Rendering — cards
// ==========================================================================

function renderCards(matches, outcome) {
  state.matches = matches;
  cardGridEl.replaceChildren();

  if (!matches.length) {
    cardsTitleEl.textContent = "No options to show yet";
    cardsCountEl.textContent = "";
    cardsNoteEl.className = "cards-note";
    cardsNoteEl.textContent =
      outcome === "needs_clarification"
        ? "Answer the question above and I'll pull up the best-fit options."
        : "Describe the care need, the days/hours you need covered, and your budget.";
    return;
  }

  const isFeatured = outcome === "featured";
  const isFallback = !isFeatured && matches[0].is_fallback;

  if (isFeatured) {
    cardsTitleEl.textContent = "A few places to start";
    cardsCountEl.textContent = `${matches.length} of ${state.totalServices || "many"} · a random sample`;
    cardsNoteEl.className = "cards-note";
    cardsNoteEl.textContent =
      "Just a sample of what's out there. Tell me about the care need, the hours you need covered and your budget above — I'll rank the ones that actually fit and draft an enquiry.";
  } else if (isFallback) {
    cardsTitleEl.textContent = "Closest options";
    cardsCountEl.textContent = `${matches.length} shown · nearest near-misses`;
    cardsNoteEl.className = "cards-note fallback";
    cardsNoteEl.textContent =
      "Nothing matched every constraint you gave, so these are the nearest near-misses — ordered by location, then budget, then schedule. No enquiry drafted; check each place can take on the care need before you contact them.";
  } else {
    cardsTitleEl.textContent = "Top respite options for you";
    cardsCountEl.textContent = `${matches.length} shown · ranked best-fit first`;
    cardsNoteEl.className = "cards-note";
    cardsNoteEl.textContent =
      "Ranked best-fit first — the same order as the numbered pins on the map. Click a card to find it on the map.";
  }

  for (const m of matches) {
    cardGridEl.appendChild(buildCard(m));
  }
}

function buildCard(m) {
  const card = document.createElement("article");
  card.className = "card";
  card.dataset.rank = m.rank;

  const emoji = TYPE_EMOJI[m.care_type] || TYPE_EMOJI.respite;
  const price =
    m.cost_per_session != null
      ? `<span class="price"><strong>~S$${Math.round(m.cost_per_session)}</strong> <span class="per">/ session</span></span>`
      : "";

  card.innerHTML = `
    <div class="photo">
      <img src="${m.photo_url}" alt="${escapeHtml(m.care_type)} respite care" loading="lazy" />
      <div class="fallback">${emoji}</div>
      <div class="rank-badge">${m.rank}</div>
      <div class="type-tag">${escapeHtml(m.care_type.replace("-", " "))}</div>
    </div>
    <div class="body">
      <div class="row1">
        <span class="name">${escapeHtml(m.name)}</span>
        <span class="area">${escapeHtml(m.area)}</span>
      </div>
      <div class="why">${escapeHtml(m.why)}</div>
      ${price}
    </div>
  `;

  const photo = card.querySelector(".photo");
  card.querySelector("img").addEventListener("error", () => photo.classList.add("img-failed"));

  card.addEventListener("mouseenter", () => setActive(m.rank, { fromCard: true, pan: false }));
  card.addEventListener("mouseleave", () => setActive(null));
  card.addEventListener("click", () => setActive(m.rank, { fromCard: true, pan: true }));

  return card;
}

// ==========================================================================
// Rendering — draft (copy-only, never sent). Rendered inside the chat panel,
// right under the transcript, so the enquiry stays with the conversation.
// ==========================================================================

function renderDraft(draft) {
  draftMountEl.replaceChildren();
  if (!draft) return;

  // Is this drafted off the "closest options" (near-miss) list? Then it doesn't
  // fit everything the caregiver asked for — say so above the draft.
  const nearMiss = (state.matches[0] || {}).is_fallback === true;
  const sub = nearMiss
    ? "This centre doesn't match everything you asked for (budget, hours or area) — send it as a first question, not a booking. This tool never contacts any service for you."
    : "Review and edit this, then send it from your own email. This tool never contacts any service for you.";

  const box = document.createElement("div");
  box.className = "draft-box";
  box.innerHTML = `
    <h3>Draft enquiry to ${escapeHtml(draft.service_name)}</h3>
    <div class="draft-sub">${sub}</div>
    <label>Subject</label>
    <input type="text" id="draftSubject" />
    <label>Message</label>
    <textarea id="draftBody"></textarea>
    <div class="draft-actions"></div>
  `;
  draftMountEl.appendChild(box);
  // Set values via .value so quotes/newlines are safe (not through innerHTML).
  const subjectEl = box.querySelector("#draftSubject");
  const bodyEl = box.querySelector("#draftBody");
  subjectEl.value = draft.subject;
  bodyEl.value = draft.body;

  // Action row: a mailto: button when the centre has a published email, else a
  // link to its web enquiry form, else a note to use the AIC hotline.
  const actions = box.querySelector(".draft-actions");
  if (draft.enquiry_email) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "mailto-btn";
    btn.textContent = `Open in your email app — to ${draft.enquiry_email}`;
    btn.addEventListener("click", () => {
      // Build the mailto from the CURRENT (possibly edited) fields.
      const url =
        `mailto:${draft.enquiry_email}` +
        `?subject=${encodeURIComponent(subjectEl.value)}` +
        `&body=${encodeURIComponent(bodyEl.value)}`;
      window.location.href = url;  // hands off to the OS mail client; caregiver still hits Send
    });
    actions.appendChild(btn);
    const hint = document.createElement("div");
    hint.className = "draft-hint";
    hint.textContent = "Opens a pre-filled draft in your mail app. Nothing is sent until you press send there.";
    actions.appendChild(hint);
  } else if (draft.enquiry_form) {
    const a = document.createElement("a");
    a.className = "mailto-btn as-link";
    a.href = draft.enquiry_form;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = "This centre takes a web enquiry form — open it";
    actions.appendChild(a);
    const hint = document.createElement("div");
    hint.className = "draft-hint";
    hint.textContent = "No public email for this centre. Copy the message above into their form.";
    actions.appendChild(hint);
  } else {
    const hint = document.createElement("div");
    hint.className = "draft-hint";
    hint.textContent = "No public email or form for this centre — enquire via the AIC hotline 1800-650-6060.";
    actions.appendChild(hint);
  }

  scrollChatToBottom();  // bring the fresh draft into view within the chat panel
}

// ==========================================================================
// Map
// ==========================================================================

function ensureMap() {
  if (state.map) return;
  if (typeof L === "undefined") {  // Leaflet CDN blocked / offline — keep the rest of the page working
    mapEmptyEl.textContent = "Map unavailable (couldn't load Leaflet). The ranked list still works.";
    return;
  }
  mapEmptyEl.hidden = true;
  mapEl.hidden = false;
  void mapEl.offsetHeight;  // force a reflow so Leaflet reads a real container size, not 0
  state.map = L.map("map", { scrollWheelZoom: true }).setView([1.3521, 103.8198], 11);
  state.tileLayer = L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom: 18, attribution: "© OpenStreetMap contributors" }
  ).addTo(state.map);
}

function clearMarkers() {
  for (const mk of state.markers) mk.remove();
  state.markers = [];
}

function renderMap(matches, opts = {}) {
  if (!matches.length) {
    clearMarkers();  // e.g. a clarifying-question turn — drop stale pins
    return;
  }
  ensureMap();
  if (!state.map) return;  // Leaflet unavailable — ensureMap left a note in the map column

  clearMarkers();

  const latlngs = [];
  for (const m of matches) {
    const icon = L.divIcon({
      className: "",
      html: `<div class="pin" data-rank="${m.rank}"><span>${m.rank}</span></div>`,
      iconSize: [30, 30],
      iconAnchor: [15, 28],
      popupAnchor: [0, -26],
    });
    const marker = L.marker([m.lat, m.lng], { icon }).addTo(state.map);
    marker.bindPopup(
      `<strong>${escapeHtml(m.name)}</strong><br/>${escapeHtml(m.area)} · ~S$${
        m.cost_per_session != null ? Math.round(m.cost_per_session) : "?"
      }/session`
    );
    marker.on("click", () => setActive(m.rank, { fromCard: false, pan: true }));
    marker.on("mouseover", () => setActive(m.rank, { fromCard: false, pan: false }));
    marker.on("mouseout", () => setActive(null));
    state.markers.push(marker);
    latlngs.push([m.lat, m.lng]);
  }

  // What to frame: all of Singapore for the initial 'featured' view, or the
  // matched pins once the caregiver has narrowed things down.
  const bounds =
    opts.fit === "singapore"
      ? L.latLngBounds(SG_BOUNDS)
      : L.latLngBounds(latlngs).pad(0.08);

  // The map container may have just been un-hidden, so Leaflet's cached size
  // can be stale (0). Recompute it a few times over the next few frames, then
  // fit the bounds.
  const settle = (n) => {
    if (!state.map) return;
    state.map.invalidateSize({ animate: false });
    state.map.fitBounds(bounds, { maxZoom: opts.fit === "singapore" ? 13 : 15 });
    if (n > 0) requestAnimationFrame(() => settle(n - 1));
  };
  requestAnimationFrame(() => settle(4));
}

// ==========================================================================
// Card <-> pin highlight sync
// ==========================================================================

function setActive(rank, opts = {}) {
  const { fromCard = false, pan = false } = opts;
  state.activeRank = rank;

  document.querySelectorAll(".card").forEach((c) => {
    c.classList.toggle("active", Number(c.dataset.rank) === rank);
  });
  document.querySelectorAll(".pin").forEach((p) => {
    p.classList.toggle("active", Number(p.dataset.rank) === rank);
  });

  if (rank == null) return;

  const marker = state.markers[rank - 1];
  if (marker && pan && state.map) {
    state.map.panTo(marker.getLatLng(), { animate: true });
    marker.openPopup();
  }
  if (!fromCard) {
    const card = document.querySelector(`.card[data-rank="${rank}"]`);
    card?.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

// ==========================================================================
// Networking
// ==========================================================================

async function send(text) {
  if (state.busy || !text.trim()) return;
  state.busy = true;
  sendBtn.disabled = true;
  inputEl.value = "";
  autosize();

  state.messages.push({ role: "user", content: text.trim() });
  renderTranscript();
  showPending();

  // 1. network — reach the server at all?
  let res;
  try {
    res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: state.messages }),
    });
  } catch (err) {
    clearPending();
    state.messages.push({
      role: "assistant",
      content: "⚠️ Could not reach the server. Is the API running?  python -m uvicorn api:app --app-dir src --port 8000  — and open this page on the same port.",
    });
    renderTranscript();
    finishTurn();
    return;
  }

  clearPending();

  // 2. read the body — JSON on success, but an error may come back as plain text
  //    (a raw 500 page). Don't let that get reported as "server unreachable".
  const raw = await res.text().catch(() => "");
  let data = null;
  try { data = raw ? JSON.parse(raw) : null; } catch { /* non-JSON error body */ }

  if (!res.ok) {
    const msg =
      (data && data.detail) ||
      (raw && raw.slice(0, 300)) ||
      `Server error ${res.status}.`;
    state.messages.push({ role: "assistant", content: `⚠️ ${res.status}: ${msg}` });
    renderTranscript();
    finishTurn();
    return;
  }
  if (!data) {
    state.messages.push({ role: "assistant", content: "⚠️ The server replied with something unexpected. Check the uvicorn console." });
    renderTranscript();
    finishTurn();
    return;
  }

  // 2. render — best-effort; a failure here still leaves the reply in place
  state.messages.push({ role: "assistant", content: data.reply || "(no response)" });
  renderTranscript();
  try {
    state.showingFeatured = false;  // the conversation now drives the cards + map
    renderCards(data.matches || [], data.outcome);
    renderMap(data.matches || []);
    renderDraft(data.draft);
  } catch (err) {
    console.error("render failed:", err);
  }
  finishTurn();
}

function finishTurn() {
  state.busy = false;
  sendBtn.disabled = false;
  inputEl.focus({ preventScroll: true });  // ready for a follow-up, but don't yank the page around
}

// ==========================================================================
// Small helpers + wiring
// ==========================================================================

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function autosize() {
  if (!inputEl.value) {           // empty -> let the CSS min-height rule stand
    inputEl.style.height = "";
    return;
  }
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
}

composerEl.addEventListener("submit", (e) => {
  e.preventDefault();
  send(inputEl.value);
});
inputEl.addEventListener("input", autosize);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send(inputEl.value);
  }
});

window.addEventListener("resize", () => {
  if (state.map) state.map.invalidateSize();  // keep tiles filling the box when the window changes
});

// ==========================================================================
// Initial view — before the caregiver has said anything: all of Singapore on
// the map, and a random sample of services on the cards.
// ==========================================================================

// Show the Singapore map straight away, independent of the /api/featured call —
// so the map is never blank on load even if that request is slow or fails.
function initMapSingapore() {
  ensureMap();
  if (!state.map) return;
  const settle = (n) => {
    if (!state.map) return;
    state.map.invalidateSize({ animate: false });
    state.map.fitBounds(L.latLngBounds(SG_BOUNDS), { maxZoom: 13 });
    if (n > 0) requestAnimationFrame(() => settle(n - 1));
  };
  requestAnimationFrame(() => settle(4));
}

async function loadFeatured() {
  initMapSingapore();  // map first, so it shows even if the sample call is slow/fails

  try {
    const [featRes, healthRes] = await Promise.all([
      fetch("/api/featured"),
      fetch("/api/health").catch(() => null),
    ]);
    if (healthRes && healthRes.ok) {
      state.totalServices = (await healthRes.json()).services || 0;
    }
    if (!featRes.ok) {
      // Older server without /api/featured — keep the Singapore map, tell the operator.
      cardsTitleEl.textContent = "Restart the server to load options";
      cardsNoteEl.className = "cards-note";
      cardsNoteEl.textContent =
        "This build adds a pre-chat sample view (GET /api/featured). Restart uvicorn to pick it up. You can still start a conversation above.";
      return;
    }
    const sample = await featRes.json();
    state.showingFeatured = true;
    renderCards(sample, "featured");
    renderMap(sample, { fit: "singapore" });
  } catch (err) {
    console.error("featured load failed:", err);  // non-fatal — the chat still works
  }
}

autosize();
loadFeatured();
