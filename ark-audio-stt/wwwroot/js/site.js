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
    const text = el.innerText || el.textContent;
    navigator.clipboard.writeText(text).then(() => {
        const original = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => (btn.textContent = original), 1400);
    });
}

// ---- Live curl generator for the API console -------------------------------
// Transcription is a multipart upload, so the generated curl references the
// chosen file name (curl reads it from disk with @) and the selected language.
function refreshCurl() {
    const box = document.getElementById("curlConsole");
    if (!box) return;
    const cfg = box.dataset;

    const fileInput = document.getElementById("File");
    let fileName = "audio.wav";
    if (fileInput && fileInput.files && fileInput.files.length > 0) {
        fileName = fileInput.files[0].name;
    }
    const language = document.getElementById("Language")?.value || "auto";

    const curl =
        `curl -X POST ${cfg.endpoint}/api/transcribe \\\n` +
        `  -H "${cfg.keyname}: ${cfg.keyvalue}" \\\n` +
        `  -F "file=@${fileName}" \\\n` +
        `  -F "language=${language}"`;
    box.textContent = curl;
}

document.addEventListener("DOMContentLoaded", function () {
    if (document.getElementById("curlConsole")) {
        ["File", "Language"].forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.addEventListener("change", refreshCurl);
        });
        refreshCurl();
    }
});
