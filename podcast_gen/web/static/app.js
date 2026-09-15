const form = document.getElementById("generate-form");
const generateBtn = document.getElementById("generate-btn");
const jobStatusEl = document.getElementById("job-status");
const jobStatusText = document.getElementById("job-status-text");
const jobProgress = document.getElementById("job-progress");
const llmBackendSelect = document.getElementById("llm_backend");
const ollamaModelRow = document.getElementById("ollama-model-row");
const episodeList = document.getElementById("episode-list");
const voicesListEl = document.getElementById("voices-list");

let pollTimer = null;

llmBackendSelect.addEventListener("change", () => {
  ollamaModelRow.hidden = llmBackendSelect.value !== "ollama";
});

async function loadVoices() {
  try {
    const res = await fetch("/api/voices");
    const data = await res.json();
    voicesListEl.textContent = data.voices.map((v) => v.id).join(", ");
  } catch (e) {
    voicesListEl.textContent = "(failed to load)";
  }
}

function statusLabel(status) {
  const labels = {
    queued: "Queued...",
    loading_source: "Loading source document...",
    writing_script: "Writing the dialogue script...",
    rendering: "Rendering audio...",
    mixing: "Mixing final episode...",
    done: "Done!",
    failed: "Failed.",
  };
  return labels[status] || status;
}

async function pollJob(jobId) {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) {
    jobStatusText.textContent = "Job not found.";
    clearInterval(pollTimer);
    generateBtn.disabled = false;
    return;
  }
  const job = await res.json();
  jobStatusText.textContent = statusLabel(job.status);

  if (job.status === "rendering" && job.lines_total > 0) {
    jobProgress.max = job.lines_total;
    jobProgress.value = job.lines_done;
    jobStatusText.textContent = `Rendering audio (${job.lines_done}/${job.lines_total} lines)...`;
  } else {
    jobProgress.removeAttribute("value");
  }

  if (job.status === "done") {
    clearInterval(pollTimer);
    generateBtn.disabled = false;
    jobStatusText.textContent = "Done! See it in the Library below.";
    loadEpisodes();
  } else if (job.status === "failed") {
    clearInterval(pollTimer);
    generateBtn.disabled = false;
    jobStatusText.textContent = `Failed: ${job.error || "unknown error"}`;
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  generateBtn.disabled = true;
  jobStatusEl.hidden = false;
  jobStatusText.textContent = "Submitting...";

  const formData = new FormData(form);
  // Only send the source field that's actually filled in.
  const sourceFile = document.getElementById("source_file").files[0];
  if (sourceFile) {
    formData.delete("source_text");
    formData.delete("source_url");
  } else if (formData.get("source_url")) {
    formData.delete("source_text");
    formData.delete("source_file");
  } else {
    formData.delete("source_url");
    formData.delete("source_file");
  }

  try {
    const res = await fetch("/api/generate", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      jobStatusText.textContent = `Error: ${err.detail || res.statusText}`;
      generateBtn.disabled = false;
      return;
    }
    const { job_id } = await res.json();
    pollTimer = setInterval(() => pollJob(job_id), 1500);
    pollJob(job_id);
  } catch (err) {
    jobStatusText.textContent = `Error: ${err}`;
    generateBtn.disabled = false;
  }
});

function formatDate(ts) {
  return new Date(ts * 1000).toLocaleString();
}

async function deleteEpisode(id) {
  if (!confirm("Delete this episode?")) return;
  await fetch(`/api/episodes/${id}`, { method: "DELETE" });
  loadEpisodes();
}

async function loadEpisodes() {
  const res = await fetch("/api/episodes");
  const episodes = await res.json();
  episodeList.innerHTML = "";
  if (episodes.length === 0) {
    episodeList.innerHTML = "<li class='empty'>No episodes yet.</li>";
    return;
  }
  for (const ep of episodes) {
    const li = document.createElement("li");
    li.className = "episode";
    const names = (ep.participants || []).map((p) => p.display_name).join(", ");
    li.innerHTML = `
      <div class="episode-title">${ep.title}</div>
      <div class="episode-meta">${formatDate(ep.created_at)} &middot; ${ep.line_count} lines &middot; ${names}</div>
      <audio controls src="/api/episodes/${ep.id}/audio"></audio>
      <div class="episode-actions">
        <a href="/api/episodes/${ep.id}/audio" download>Download</a>
        <button data-id="${ep.id}" class="delete-btn">Delete</button>
      </div>
    `;
    li.querySelector(".delete-btn").addEventListener("click", () => deleteEpisode(ep.id));
    episodeList.appendChild(li);
  }
}

document.getElementById("refresh-btn").addEventListener("click", loadEpisodes);

loadVoices();
loadEpisodes();
