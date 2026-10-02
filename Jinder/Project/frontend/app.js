//  CONFIG 
// FastAPI base URL (change  to  real host/port)
// ==== Backend connection config ====
//                  "http://<EC2_PUBLIC_IP>:8000"
// Same origin as the HTML page
const API_BASE_URL = ""; // or window.location.origin  

// These match FastAPI routes in api.py
const ENDPOINTS = {
  health: "/healthcheck",
  login: "/login",
  signup: "/register",
  profile: "/profile",
  match: "/match"
};

// Offline demo mode. Lets a reviewer click through the UI without AWS
// credentials or a running backend. Entered by the explicit "Explore the demo"
// buttons, not by typing a password -- there are deliberately no credentials
// committed here.
//
// This only populates client-side state. It is not an authentication path:
// the API never sees it and never trusts it.
const DEMO_ROLES = {
  user: { email: "demo-user@example.com", role: "user" },
  admin: { email: "demo-admin@example.com", role: "admin" }
};


//  STATE 

const state = {
  currentUser: null,
  token: null,      // not used yet ( doesn't issue JWTs, but we keep it for future)
  matches: [],
  resume: null,
  adminUsers: []
};

//  API HELPERS 

async function apiJson(path, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}),
    ...(options.headers || {})
  };

  const res = await fetch(API_BASE_URL + path, {
    ...options,
    headers
  });

  let data = null;
  try {
    data = await res.json();
  } catch {
    // no JSON body
  }

  if (!res.ok) {
    const msg = data?.detail || data?.message || data?.error || `Request failed (${res.status})`;
    throw new Error(msg);
  }

  return data;
}

async function apiForm(path, formData, options = {}) {
  const headers = {
    ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}),
    ...(options.headers || {})
  };

  const url = API_BASE_URL + path;
  const res = await fetch(url, {
    method: "POST",
    body: formData,
    headers
  });

  let data = null;
  try {
    data = await res.json();
  } catch {
    // no JSON body
  }

  if (!res.ok) {
    const msg = data?.detail || data?.message || data?.error || `Request failed (${res.status})`;
    throw new Error(msg);
  }

  return data;
}

// =================== BOOTSTRAP ===================

document.addEventListener("DOMContentLoaded", () => {
  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  setupViewNavigation();
  setupAuthTabs();
  setupAuthForms();
  setupPasswordToggles();
  setupResumeUploader();
  setupProfileForms();
  setupAdminActions();
  updateUIForAuth();
});

//  VIEW NAVIGATION 

function setupViewNavigation() {
  const navButtons = document.querySelectorAll("[data-view]");
  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const view = btn.getAttribute("data-view");

      if ((view === "dashboard" || view === "profile") && !state.currentUser) {
        toast("Please log in first.", "error");
        showView("landing");
        return;
      }

      if (
        view === "admin" &&
        (!state.currentUser || state.currentUser.role !== "admin")
      ) {
        toast("Admin access only.", "error");
        showView("landing");
        return;
      }

      showView(view);
    });
  });
}

function showView(viewId) {
  const views = document.querySelectorAll(".view");
  views.forEach((v) => v.classList.remove("view-active"));

  const target = document.getElementById(`view-${viewId}`);
  if (target) target.classList.add("view-active");

  const navButtons = document.querySelectorAll(".nav-link");
  navButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-view") === viewId);
  });
}

//  AUTH TABS (LOGIN / SIGNUP) 

function setupAuthTabs() {
  const tabs = document.querySelectorAll("[data-auth-tab]");
  const loginForm = document.getElementById("loginForm");
  const signupForm = document.getElementById("signupForm");

  const openLoginBtn = document.getElementById("openLogin");
  const openSignupBtn = document.getElementById("openSignup");

  function activateTab(mode) {
    tabs.forEach((t) =>
      t.classList.toggle("active", t.getAttribute("data-auth-tab") === mode)
    );

    if (mode === "login") {
      loginForm.classList.remove("hidden");
      signupForm.classList.add("hidden");
    } else {
      signupForm.classList.remove("hidden");
      loginForm.classList.add("hidden");
    }

    clearFieldErrors();
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      activateTab(tab.getAttribute("data-auth-tab"));
    });
  });

  if (openLoginBtn) {
    openLoginBtn.addEventListener("click", () => activateTab("login"));
  }
  if (openSignupBtn) {
    openSignupBtn.addEventListener("click", () => activateTab("signup"));
  }

  document.querySelectorAll("[data-demo-role]").forEach((btn) => {
    btn.addEventListener("click", () => enterDemoMode(btn.dataset.demoRole));
  });
}

//  DEMO MODE

function enterDemoMode(roleName) {
  const demo = DEMO_ROLES[roleName];
  if (!demo) return;

  state.token = null;
  state.currentUser = {
    email: demo.email,
    role: demo.role,
    lastLogin: new Date(),
    resumesUploaded: state.resume ? 1 : 0
  };

  toast(`Exploring the demo as ${demo.role}. No backend calls are made.`, "success");
  updateUIForAuth();
  renderMatches();
  syncProfileSection();
  showView(demo.role === "admin" ? "admin" : "dashboard");
}

//  AUTH FORMS

function setupAuthForms() {
  const loginForm = document.getElementById("loginForm");
  const signupForm = document.getElementById("signupForm");
  const logoutBtn = document.getElementById("logoutBtn");

  if (loginForm) {
    loginForm.addEventListener("submit", (e) => {
      e.preventDefault();
      handleLogin();
    });
  }

  if (signupForm) {
    signupForm.addEventListener("submit", (e) => {
      e.preventDefault();
      handleSignup();
    });
  }

  if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
      state.currentUser = null;
      state.token = null;
      state.matches = [];
      state.resume = null;
      toast("Logged out.", "success");
      updateUIForAuth();
      showView("landing");
    });
  }
}

async function handleLogin() {
  clearFieldErrors();

  const email = document.getElementById("loginEmail").value.trim();
  const password = document.getElementById("loginPassword").value;

  let valid = true;
  if (!isValidEmail(email)) {
    setFieldError("loginEmail", "Please enter a valid email.");
    valid = false;
  }
  if (!isValidPassword(password)) {
    setFieldError(
      "loginPassword",
      "Password must be at least 6 characters with letters and numbers."
    );
    valid = false;
  }
  if (!valid) return;

  // Real API login (POST /login). Demo mode is entered through
  // enterDemoMode() instead, so no credentials are checked client-side.
  try {
    const data = await apiJson(ENDPOINTS.login, {
      method: "POST",
      body: JSON.stringify({ email, password })
    });

    // FastAPI /login returns: { "user": user }
    // We don't get a JWT, so token stays null for now.
    state.token = null;
    const user = data.user || { email, role: "user" };

    state.currentUser = {
      email: user.email,
      role: user.role || "user",
      lastLogin: new Date(),
      resumesUploaded: user.resumesUploaded || 0
    };

    toast("Login successful. Welcome to Jinder!", "success");
    updateUIForAuth();
    renderMatches();
    syncProfileSection();
    showView(state.currentUser.role === "admin" ? "admin" : "dashboard");
  } catch (err) {
    setFieldError("loginPassword", err.message);
    toast("Login failed: " + err.message, "error");
  }
}

async function handleSignup() {
  clearFieldErrors();

  const email = document.getElementById("signupEmail").value.trim();
  const password = document.getElementById("signupPassword").value;
  const passwordConfirm = document.getElementById(
    "signupPasswordConfirm"
  ).value;

  let valid = true;

  if (!isValidEmail(email)) {
    setFieldError("signupEmail", "Please enter a valid email.");
    valid = false;
  }
  if (!isValidPassword(password)) {
    setFieldError(
      "signupPassword",
      "Password must be at least 6 characters with letters and numbers."
    );
    valid = false;
  }
  if (password !== passwordConfirm) {
    setFieldError(
      "signupPasswordConfirm",
      "Passwords do not match."
    );
    valid = false;
  }
  if (!valid) return;

  try {
    // FastAPI /register expects:
    // { email, password, default_title?, default_location?, remote_pref? }
    const body = {
      email,
      password,
      default_title: null,
      default_location: null,
      remote_pref: null
    };

    const data = await apiJson(ENDPOINTS.signup, {
      method: "POST",
      body: JSON.stringify(body)
    });

    const user = data.user || { email, role: "user" };

    state.token = null;
    state.currentUser = {
      email: user.email,
      role: user.role || "user",
      lastLogin: new Date(),
      resumesUploaded: user.resumesUploaded || 0
    };

    toast("Account created. You are now logged in.", "success");
    updateUIForAuth();
    showView(state.currentUser.role === "admin" ? "admin" : "dashboard");
  } catch (err) {
    toast("Sign up failed: " + err.message, "error");
  }
}

//  PASSWORD TOGGLE 

function setupPasswordToggles() {
  const toggles = document.querySelectorAll("[data-toggle-password]");
  toggles.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-toggle-password");
      const input = document.getElementById(targetId);
      if (!input) return;

      if (input.type === "password") {
        input.type = "text";
        btn.textContent = "Hide";
      } else {
        input.type = "password";
        btn.textContent = "Show";
      }
    });
  });
}

//  VALIDATION HELPERS 

function isValidEmail(email) {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
}

function isValidPassword(password) {
  if (!password || password.length < 6) return false;
  const hasLetter = /[a-zA-Z]/.test(password);
  const hasNumber = /\d/.test(password);
  return hasLetter && hasNumber;
}

function setFieldError(fieldId, message) {
  const errorEl = document.querySelector(`[data-error-for="${fieldId}"]`);
  if (errorEl) errorEl.textContent = message;
}

function clearFieldErrors() {
  document
    .querySelectorAll(".field-error")
    .forEach((el) => (el.textContent = ""));
}

//  RESUME UPLOAD + MATCHES 

function setupResumeUploader() {
  const dropzone = document.getElementById("resumeDropzone");
  const fileInput = document.getElementById("resumeInput");
  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dropzone-hover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dropzone-hover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dropzone-hover");
    const file = e.dataTransfer.files[0];
    handleResumeFile(file);
  });

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    handleResumeFile(file);
  });
}

async function handleResumeFile(file) {
  if (!file) return;

  if (!file.name.toLowerCase().endsWith(".pdf")) {
    toast("Only PDF files are allowed.", "error");
    return;
  }

  if (!state.currentUser) {
    toast("Please log in before uploading a resume.", "error");
    return;
  }

  const formData = new FormData();
  // IMPORTANT: FastAPI /match parameter is named `file`, so key must be "file"
  formData.append("file", file);

  try {
    // Optional: add top_k as a query param (FastAPI default = 10)
    const urlWithQuery = `${ENDPOINTS.match}?top_k=10`;
    const data = await apiForm(urlWithQuery, formData);

    // FastAPI /match returns: { "match_results": [...] }
    const results = data.match_results || [];

    state.resume = {
      name: file.name,
      uploadedAt: new Date()
    };

    state.matches = results;

    if (state.currentUser) {
      state.currentUser.resumesUploaded =
        (state.currentUser.resumesUploaded || 0) + 1;
    }

    renderResumeMeta();
    renderMatches();
    syncProfileSection();
    syncAdminUsers();
    toast("Resume uploaded and matches refreshed.", "success");
  } catch (err) {
    toast("Failed to upload resume: " + err.message, "error");
  }
}

function renderResumeMeta() {
  const meta = document.getElementById("resumeMeta");
  const nameEl = document.getElementById("resumeFilename");
  const dateEl = document.getElementById("resumeDate");

  if (!meta || !state.resume) {
    if (meta) meta.classList.add("hidden");
    return;
  }

  nameEl.textContent = state.resume.name;
  dateEl.textContent = `Uploaded on ${state.resume.uploadedAt.toLocaleString()}`;
  meta.classList.remove("hidden");
}

function renderMatches() {
  const container = document.getElementById("matchesContainer");
  const hint = document.getElementById("matchHint");
  const countPill = document.getElementById("matchCountPill");

  if (!container) return;

  container.innerHTML = "";

  if (!state.matches || state.matches.length === 0) {
    if (hint) {
      hint.textContent =
        "Upload a resume to generate matches (or check your backend).";
    }
    if (countPill) countPill.textContent = "0 matches";
    return;
  }

  if (hint) {
    hint.textContent = "These matches are returned by the backend API.";
  }

  if (countPill) {
    countPill.textContent = `${state.matches.length} matches`;
  }

  // We don't know exact shape of each `match_result` from JobMatcher,
  // so we try to map common fields and provide fallbacks.
  state.matches.forEach((m) => {
    const card = document.createElement("div");
    card.className = "match-card";

    const titleRow = document.createElement("div");
    titleRow.className = "match-title-row";

    // title: try a few likely fields
    const titleText =
      m.title ||
      m.job_title ||
      m.role ||
      (m.metadata && (m.metadata.title || m.metadata.job_title)) ||
      "Untitled";

    const title = document.createElement("div");
    title.className = "match-title";
    title.textContent = titleText;

    const score = document.createElement("div");
    score.className = "match-score";
    if (typeof m.score === "number") {
      score.textContent = `${(m.score * 100).toFixed(1)}% match`;
    } else if (m.scoreText) {
      score.textContent = m.scoreText;
    } else {
      score.textContent = "";
    }

    titleRow.appendChild(title);
    titleRow.appendChild(score);

    const meta = document.createElement("div");
    meta.className = "match-meta";
    const company =
      m.company ||
      m.org ||
      (m.metadata && (m.metadata.company || m.metadata.organization)) ||
      "Unknown org";
    const type =
      m.type ||
      m.kind ||
      (m.source && m.source.toUpperCase && m.source.toUpperCase()) ||
      "Opportunity";
    const source =
      m.source ||
      (m.metadata && m.metadata.source) ||
      "Backend";

    meta.innerHTML = `
      <span>${company}</span>
      <span>• ${type}</span>
      <span>• Source: ${source}</span>
    `;

    const tags = document.createElement("div");
    tags.className = "match-tags";
    const tagList =
      m.tags ||
      m.skills ||
      (m.metadata && (m.metadata.tags || m.metadata.skills)) ||
      [];
    tagList.forEach((t) => {
      const tag = document.createElement("span");
      tag.className = "match-tag";
      tag.textContent = t;
      tags.appendChild(tag);
    });

    card.appendChild(titleRow);
    card.appendChild(meta);
    card.appendChild(tags);
    container.appendChild(card);
  });
}

//  PROFILE 

function setupProfileForms() {
  const emailForm = document.getElementById("emailForm");
  const passwordForm = document.getElementById("passwordForm");

  // NOTE:  FastAPI /profile endpoints are email-based preferences.
  // For now, we'll keep these forms local only

  if (emailForm) {
    emailForm.addEventListener("submit", (e) => {
      e.preventDefault();
      clearFieldErrors();

      if (!state.currentUser) {
        toast("You must be logged in to change your email.", "error");
        return;
      }

      const email = document.getElementById("profileEmail").value.trim();
      if (!isValidEmail(email)) {
        setFieldError("profileEmail", "Please enter a valid email.");
        return;
      }

      state.currentUser.email = email;
      syncAdminUsers();
      toast("Email updated (local only).", "success");
      updateUIForAuth();
    });
  }

  if (passwordForm) {
    passwordForm.addEventListener("submit", (e) => {
      e.preventDefault();
      clearFieldErrors();

      const pwd = document.getElementById("profilePassword").value;
      const pwd2 = document.getElementById(
        "profilePasswordConfirm"
      ).value;

      if (!pwd || !pwd2) {
        setFieldError(
          "profilePassword",
          "Please enter a new password."
        );
        return;
      }

      if (!isValidPassword(pwd)) {
        setFieldError(
          "profilePassword",
          "Password must be at least 6 characters with letters and numbers."
        );
        return;
      }

      if (pwd !== pwd2) {
        setFieldError(
          "profilePasswordConfirm",
          "Passwords do not match."
        );
        return;
      }

      //  backend doesn't expose a password-change endpoint yet.
      toast("Password updated (frontend demo only).", "success");
      passwordForm.reset();
    });
  }
}

function syncProfileSection() {
  const emailInput = document.getElementById("profileEmail");
  const resumeBlock = document.getElementById("profileResumeBlock");
  const profileMatches = document.getElementById("profileMatches");

  if (state.currentUser && emailInput) {
    emailInput.value = state.currentUser.email;
  }

  if (resumeBlock) {
    if (state.resume) {
      resumeBlock.classList.remove("text-muted");
      resumeBlock.innerHTML = `
        <p><strong>Resume:</strong> ${state.resume.name}</p>
        <p class="text-muted">
          Uploaded on ${state.resume.uploadedAt.toLocaleString()}
        </p>
      `;
    } else {
      resumeBlock.classList.add("text-muted");
      resumeBlock.textContent =
        "No resume uploaded yet. Upload one from the Matches page.";
    }
  }

  if (profileMatches) {
    if (state.matches && state.matches.length > 0) {
      profileMatches.classList.remove("text-muted");
      profileMatches.innerHTML = "";
      state.matches.slice(0, 4).forEach((m) => {
        const chip = document.createElement("div");
        chip.className = "match-chip";
        const title = m.title || m.job_title || m.role || "Untitled";
        const company =
          m.company ||
          m.org ||
          (m.metadata && (m.metadata.company || m.metadata.organization)) ||
          "Org";
        chip.innerHTML = `<strong>${title}</strong> · ${company}`;
        profileMatches.appendChild(chip);
      });
    } else {
      profileMatches.classList.add("text-muted");
      profileMatches.textContent =
        "No matches yet. Upload a resume to see suggestions.";
    }
  }
}

//  ADMIN 

function setupAdminActions() {
  syncAdminUsers();

  const buttons = document.querySelectorAll("[data-maintenance]");
  const log = document.getElementById("maintenanceLog");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!log) return;
      const action = btn.getAttribute("data-maintenance");
      const now = new Date().toLocaleTimeString();
      let label;
      switch (action) {
        case "reindex":
          label = "Rebuilt search indexes";
          break;
        case "cleanup":
          label = "Ran stale-data cleanup";
          break;
        case "backup":
          label = "Triggered backup snapshot";
          break;
        case "health":
          label = "Checked DB health";
          break;
        default:
          label = "Performed maintenance";
      }

      const entry = document.createElement("div");
      entry.textContent = `[${now}] ${label}`;
      log.prepend(entry);
      toast(`${label} (demo only).`, "success");
    });
  });
}

function syncAdminUsers() {
  const tableBody = document.querySelector("#adminUsersTable tbody");
  if (!tableBody) return;

  const baseUsers = [
    {
      email: "student1@example.com",
      role: "user",
      lastLogin: "2025-11-01 14:23",
      resumesUploaded: 1,
      matches: 5
    },
    {
      email: "professor@example.edu",
      role: "user",
      lastLogin: "2025-11-05 09:10",
      resumesUploaded: 2,
      matches: 8
    },
    {
      email: "admin@jinder.cloud",
      role: "admin",
      lastLogin: "2025-11-10 18:45",
      resumesUploaded: 0,
      matches: 0
    }
  ];

  const current = state.currentUser
    ? [
        {
          email: state.currentUser.email,
          role: state.currentUser.role,
          lastLogin: state.currentUser.lastLogin.toLocaleString(),
          resumesUploaded: state.currentUser.resumesUploaded || 0,
          matches: state.matches.length || 0
        }
      ]
    : [];

  const users = mergeUsers(baseUsers, current);
  state.adminUsers = users;

  tableBody.innerHTML = "";
  users.forEach((u) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${u.email}</td>
      <td>${u.role}</td>
      <td>${u.lastLogin}</td>
      <td>${u.resumesUploaded}</td>
      <td>${u.matches}</td>
    `;
    tableBody.appendChild(tr);
  });
}

function mergeUsers(base, current) {
  const byEmail = new Map();
  base.forEach((u) => byEmail.set(u.email, u));
  current.forEach((u) => byEmail.set(u.email, u));
  return Array.from(byEmail.values());
}

//  AUTH STATE UI 

function updateUIForAuth() {
  const label = document.getElementById("currentUserLabel");
  const logoutBtn = document.getElementById("logoutBtn");
  const authRequired = document.querySelectorAll(".requires-auth");
  const adminRequired = document.querySelectorAll(".requires-admin");

  const isLoggedIn = !!state.currentUser;
  const isAdmin = isLoggedIn && state.currentUser.role === "admin";

  if (label) {
    label.textContent = isLoggedIn ? state.currentUser.email : "Guest";
  }

  if (logoutBtn) {
    logoutBtn.classList.toggle("hidden", !isLoggedIn);
  }

  authRequired.forEach((el) =>
    el.classList.toggle("hidden", !isLoggedIn)
  );
  adminRequired.forEach((el) =>
    el.classList.toggle("hidden", !isAdmin)
  );

  syncProfileSection();
  syncAdminUsers();
}

//  TOAST 

let toastTimeout = null;

function toast(message, type = "success") {
  const toastEl = document.getElementById("toast");
  if (!toastEl) return;

  toastEl.textContent = message;
  toastEl.className = `toast show ${type}`;

  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toastEl.classList.remove("show");
  }, 2800);
}