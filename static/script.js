let datasetColumns = [];
let preprocessingDone = false;

// -------------------- UPLOAD --------------------
async function uploadFile() {
    preprocessingDone = false;
    document.getElementById("previewContainer").innerHTML = "";
    document.getElementById("trainBtn").classList.add("hidden");
    document.getElementById("resultsContainer").style.display = "none";

    const file = document.getElementById("fileInput").files[0];
    if (!file) { alert("Please select a file first!"); return; }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res  = await fetch("http://127.0.0.1:5000/upload", { method: "POST", body: formData });
        const data = await res.json();

        datasetColumns = data.columns;
        document.getElementById("configCard").classList.remove("hidden");

        // Original dataset preview
        document.getElementById("previewContainer").innerHTML = `
            <div class="card">
                <div class="card-title">Original Dataset Preview</div>
                <div class="table-wrap">
                    <table>
                        <thead><tr>${data.columns.map(c => `<th>${c}</th>`).join("")}</tr></thead>
                        <tbody>${data.rows.map(row =>
                            `<tr>${row.map(cell => `<td>${cell ?? ""}</td>`).join("")}</tr>`
                        ).join("")}</tbody>
                    </table>
                </div>
            </div>`;

        showOrdinalSelector(data.categorical_cols);

    } catch (err) {
        console.error(err);
        alert("Upload failed. Make sure the backend is running.");
    }
}


// -------------------- ORDINAL SELECT --------------------
function showOrdinalSelector(categoricalCols) {
    const select = document.getElementById("ordinalCols");
    select.innerHTML = "";
    categoricalCols.forEach(col => {
        const opt = document.createElement("option");
        opt.value = col;
        opt.textContent = col;
        select.appendChild(opt);
    });
}


// -------------------- PREPROCESS --------------------
async function runPreprocessing() {
    const selectedOrdinal = Array.from(document.getElementById("ordinalCols").selectedOptions).map(o => o.value);
    const task   = document.getElementById("mlTask").value;
    const target = document.getElementById("targetCol").value;

    try {
        const res  = await fetch("http://127.0.0.1:5000/preprocess", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ordinal_cols: selectedOrdinal, task, target_column: target })
        });
        const data = await res.json();

        if (data.error) { alert("Error: " + data.error); return; }

        showPreprocessingResults(data);
        showTrainButton();
        preprocessingDone = true;

    } catch (err) {
        console.error(err);
        alert("Preprocessing failed.");
    }
}


// -------------------- PREPROCESSING RESULTS --------------------
function showPreprocessingResults(data) {
    const s = data.steps;

    // Build summary items
    const summaryItems = [
        { label: "High Cardinality Removed",  value: (s.removed_high_cardinality_cols || []).join(", ") || "None" },
        { label: "ID Columns Removed",         value: (s.removed_id_columns || []).join(", ") || "None" },
        { label: "Constant Columns Removed",   value: (s.removed_constant_columns || []).join(", ") || "None" },
        { label: "High Missing Removed",        value: (s.removed_high_missing_columns || []).join(", ") || "None" },
        { label: "Numeric Imputation",          value: s.numeric_imputation },
        { label: "Categorical Imputation",      value: s.categorical_imputation },
        { label: "Ordinal Encoding",            value: (s.ordinal_encoding || []).join(", ") || "None" },
        { label: "Frequency Encoding",          value: (s.frequency_encoding || []).join(", ") || "None" },
        { label: "Scaling",                     value: s.scaling },
        { label: "Imbalance Handling",          value: s.imbalance_handling },
    ];

    // Append to preview container
    document.getElementById("previewContainer").innerHTML += `
        <div class="card">
            <div class="card-title">Preprocessing Summary</div>
            <div class="summary-grid">
                ${summaryItems.map(item => `
                    <div class="summary-item">
                        <div class="s-label">${item.label}</div>
                        <div>${item.value}</div>
                    </div>`).join("")}
            </div>
        </div>

        <div class="card">
            <div class="card-title">Processed Dataset Preview</div>
            <div class="table-wrap">
                <table>
                    <thead><tr>${data.columns.map(c => `<th>${c}</th>`).join("")}</tr></thead>
                    <tbody>${data.rows.map(row =>
                        `<tr>${row.map(cell => `<td>${cell ?? ""}</td>`).join("")}</tr>`
                    ).join("")}</tbody>
                </table>
            </div>
        </div>`;
}


// -------------------- TASK CHANGE --------------------
function handleTaskChange() {
    const task        = document.getElementById("mlTask").value;
    const targetField = document.getElementById("targetField");
    const targetSel   = document.getElementById("targetCol");

    targetSel.innerHTML = "";

    if (task === "classification" || task === "regression") {
        datasetColumns.forEach(col => {
            const opt = document.createElement("option");
            opt.value = col;
            opt.textContent = col;
            targetSel.appendChild(opt);
        });
        targetField.classList.remove("hidden");
    } else {
        targetField.classList.add("hidden");
    }

    if (!preprocessingDone) {
        document.getElementById("trainBtn").classList.add("hidden");
        document.getElementById("resultsContainer").style.display = "none";
    }
}
