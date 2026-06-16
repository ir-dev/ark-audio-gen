// ---- Theme (dark default, persisted) ---------------------------------------
(function () {
    const KEY = "ark-theme";
    const root = document.documentElement;
    const saved = localStorage.getItem(KEY) || "dark";
    root.setAttribute("data-theme", saved);

    window.toggleTheme = function () {
        const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        localStorage.setItem(KEY, next);
        updateToggleIcon();
    };

    function updateToggleIcon() {
        const btn = document.getElementById("themeToggle");
        if (btn) btn.textContent = root.getAttribute("data-theme") === "dark" ? "☀️" : "🌙";
    }
    document.addEventListener("DOMContentLoaded", updateToggleIcon);
})();

// ---- Copy-to-clipboard buttons ---------------------------------------------
function copyFrom(targetId, btn) {
    const el = document.getElementById(targetId);
    if (!el) return;
    // Ignore the embedded copy button's own label.
    const clone = el.cloneNode(true);
    clone.querySelectorAll(".copy-btn").forEach((b) => b.remove());
    const text = clone.innerText || clone.textContent;
    navigator.clipboard.writeText(text).then(() => {
        const original = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => (btn.textContent = original), 1400);
    });
}

// ---- Live curl generator for the API console -------------------------------
// Summarize takes a JSON body, so the generated curl JSON-encodes the text the
// user typed and the chosen maxSentences value.
function refreshCurl() {
    const box = document.getElementById("curlConsole");
    if (!box) return;
    const cfg = box.dataset;

    let text = document.getElementById("Text")?.value || "";
    if (!text.trim()) text = "Your text goes here.";
    // Keep the preview readable: collapse whitespace and cap length.
    text = text.replace(/\s+/g, " ").trim();
    if (text.length > 400) text = text.slice(0, 400) + "…";

    let max = parseInt(document.getElementById("MaxSentences")?.value || "3", 10);
    if (isNaN(max) || max < 1) max = 3;
    if (max > 10) max = 10;

    // Use the first ticked engine for the API model field (the API runs one model per call).
    const checked = document.querySelector('input[name="SelectedEngines"]:checked');
    const model = checked ? checked.value : (cfg.model || "lexical");

    const body = JSON.stringify({ text: text, maxSentences: max, model: model });
    // Escape single quotes for the shell-single-quoted -d argument.
    const safeBody = body.replace(/'/g, "'\\''");

    const curl =
        `curl -X POST ${cfg.endpoint}/api/summarize \\\n` +
        `  -H "${cfg.keyname}: ${cfg.keyvalue}" \\\n` +
        `  -H "Content-Type: application/json" \\\n` +
        `  -d '${safeBody}'`;
    box.textContent = curl;
}

document.addEventListener("DOMContentLoaded", function () {
    if (document.getElementById("curlConsole")) {
        ["Text", "MaxSentences"].forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.addEventListener("input", refreshCurl);
        });
        document.querySelectorAll('input[name="SelectedEngines"]').forEach((el) =>
            el.addEventListener("change", refreshCurl));
        refreshCurl();
    }
});
