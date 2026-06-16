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
function refreshCurl() {
    const box = document.getElementById("curlConsole");
    if (!box) return;
    const cfg = box.dataset;
    const text = (document.getElementById("Text")?.value || "").trim() || "Hello world";
    const source = document.getElementById("SourceLanguage")?.value || "auto";
    const target = document.getElementById("TargetLanguage")?.value || "eng_Latn";

    const payload = JSON.stringify({ text, source, target });
    const curl =
        `curl -X POST ${cfg.endpoint}/api/translate \\\n` +
        `  -H "Content-Type: application/json" \\\n` +
        `  -H "${cfg.keyname}: ${cfg.keyvalue}" \\\n` +
        `  -d '${payload.replace(/'/g, "'\\''")}'`;
    box.textContent = curl;
}

document.addEventListener("DOMContentLoaded", function () {
    if (document.getElementById("curlConsole")) {
        ["Text", "SourceLanguage", "TargetLanguage"].forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.addEventListener("input", refreshCurl);
        });
        refreshCurl();
    }
});
