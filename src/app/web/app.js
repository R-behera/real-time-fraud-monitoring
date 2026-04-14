
const pretty = (value) => JSON.stringify(value, null, 2);

async function postJson(endpoint, body, output) {
  output.textContent = "Loading...";
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    output.textContent = pretty(payload);
  } catch (error) {
    output.textContent = pretty({ error: String(error) });
  }
}

function buildPayload(endpoint, inputMode, rawValue) {
  if (inputMode === "json") {
    return JSON.parse(rawValue);
  }
  if (endpoint === "/query") {
    return { query: rawValue };
  }
  if (endpoint === "/recommend") {
    return { prompt: rawValue };
  }
  return { value: rawValue };
}

document.querySelectorAll(".js-module").forEach((moduleNode) => {
  const button = moduleNode.querySelector(".js-run");
  const output = moduleNode.querySelector(".js-output");
  const input = moduleNode.querySelector(".js-input");
  button.addEventListener("click", async () => {
    try {
      const payload = buildPayload(button.dataset.endpoint, button.dataset.inputMode, input.value);
      await postJson(button.dataset.endpoint, payload, output);
    } catch (error) {
      output.textContent = pretty({ error: String(error) });
    }
  });
});

fetch("/bootstrap")
  .then((response) => response.json())
  .then((payload) => {
    const badge = document.querySelector("[data-health]");
    if (badge) {
      badge.textContent = `${payload.project.title} ready`;
    }
  })
  .catch(() => {
    const badge = document.querySelector("[data-health]");
    if (badge) {
      badge.textContent = "Application ready";
    }
  });
