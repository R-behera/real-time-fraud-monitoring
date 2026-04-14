
const bootstrapNode = document.getElementById("bootstrap-data");
const fallback = bootstrapNode ? JSON.parse(bootstrapNode.textContent) : {};
const pretty = (value) => JSON.stringify(value, null, 2);

const els = {
  title: document.getElementById("project-title"),
  summary: document.getElementById("project-summary"),
  tags: document.getElementById("project-tags"),
  scoreInput: document.getElementById("score-input"),
  analyzeInput: document.getElementById("analyze-input"),
  queryInput: document.getElementById("query-input"),
  scoreOutput: document.getElementById("score-output"),
  analyzeOutput: document.getElementById("analyze-output"),
  queryOutput: document.getElementById("query-output"),
  loadDefaults: document.getElementById("load-defaults"),
};

function hydrate(payload) {
  const project = payload.project || fallback.project || {};
  els.title.textContent = project.title || "Portfolio Application";
  els.summary.textContent = project.summary || "";
  const tags = project.tags || [];
  els.tags.innerHTML = tags.map((tag) => `<li>${tag}</li>`).join("");
  els.scoreInput.value = pretty({ features: payload.sample_features || fallback.sample_features || {} });
  els.analyzeInput.value = pretty({
    metrics: payload.sample_metrics || fallback.sample_metrics || {},
    baseline: Object.fromEntries(
      Object.entries(payload.sample_metrics || fallback.sample_metrics || {}).map(([key, value]) => [key, Number(value) * 0.92])
    ),
  });
  els.queryInput.value = payload.sample_query || fallback.sample_query || "What is the top priority recommendation?";
  els.scoreOutput.textContent = pretty({
    tip: "Run the scoring action to inspect a live API response.",
    endpoints: ["/score", "/analyze", "/query"],
  });
  els.analyzeOutput.textContent = pretty({
    tip: "Use metric comparison to create a fast stakeholder summary.",
  });
  els.queryOutput.textContent = pretty({
    tip: "Ask a grounded question and inspect the supporting contexts.",
    contexts: payload.knowledge_base || fallback.knowledge_base || [],
  });
}

async function safeJsonFetch(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

async function postJson(url, body, target) {
  target.textContent = "Loading...";
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    target.textContent = pretty(payload);
  } catch (error) {
    target.textContent = pretty({ error: String(error) });
  }
}

document.getElementById("run-score").addEventListener("click", async () => {
  const body = JSON.parse(els.scoreInput.value);
  await postJson("/score", body, els.scoreOutput);
});

document.getElementById("run-analyze").addEventListener("click", async () => {
  const body = JSON.parse(els.analyzeInput.value);
  await postJson("/analyze", body, els.analyzeOutput);
});

document.getElementById("run-query").addEventListener("click", async () => {
  const body = { query: els.queryInput.value };
  await postJson("/query", body, els.queryOutput);
});

els.loadDefaults.addEventListener("click", () => hydrate(fallback));

safeJsonFetch("/bootstrap")
  .then((payload) => hydrate({ ...fallback, ...payload }))
  .catch(() => hydrate(fallback));
