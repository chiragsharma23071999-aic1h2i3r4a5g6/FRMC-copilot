(() => {
    const form = document.getElementById("upload-form");
    const filesInput = document.getElementById("upload-files");
    const folderInput = document.getElementById("upload-folder");
    const selection = document.getElementById("upload-selection");
    const status = document.getElementById("upload-status");
    const list = document.getElementById("uploaded-documents");
    const refresh = document.getElementById("refresh-documents");
    const limits = window.uploadLimits;
    const supported = /\.(txt|csv|xlsx|xls|pdf|docx)$/i;
    let busy = false;
    const sizeLabel = bytes => `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    const selectedFiles = () => [...filesInput.files,
        ...Array.from(folderInput.files).filter(file => supported.test(file.name))];

    function updateSelection() {
        const files = selectedFiles();
        const skipped = Array.from(folderInput.files).filter(file => !supported.test(file.name)).length;
        selection.textContent = `${files.length} document(s) selected (${sizeLabel(files.reduce((sum, file) => sum + file.size, 0))}).` +
            (skipped ? ` ${skipped} unsupported folder file(s) skipped.` : "");
    }
    filesInput.addEventListener("change", updateSelection);
    folderInput.addEventListener("change", updateSelection);

    function setBusy(value) {
        busy = value;
        form.querySelectorAll("input, button").forEach(element => { element.disabled = value; });
        list.querySelectorAll("button").forEach(element => { element.disabled = value; });
        refresh.disabled = value;
    }

    async function responseData(response) {
        let data;
        try { data = await response.json(); }
        catch { throw new Error(response.status === 413 ? "Upload exceeds the server's request limit." : "The server returned an unreadable response. Refresh files to check the result."); }
        if (!response.ok) throw new Error([data.error || "Request failed.",
            ...(data.rejected || []).map(item => `${item.filename}: ${item.reason}`)].join("\n"));
        return data;
    }

    async function loadDocuments() {
        const data = await responseData(await fetch("/documents"));
        list.replaceChildren();
        if (!data.files.length) {
            const empty = document.createElement("li");
            empty.textContent = "No uploaded documents. Choose files or a folder above to get started.";
            list.append(empty);
        }
        for (const file of data.files) {
            const row = document.createElement("li");
            row.style.marginBottom = "8px";
            row.style.overflowWrap = "anywhere";
            row.append(document.createTextNode(`${file.filename} (${sizeLabel(file.size)}) `));
            const remove = document.createElement("button");
            remove.type = "button";
            remove.textContent = "Remove";
            remove.setAttribute("aria-label", `Remove ${file.filename}`);
            remove.disabled = busy;
            remove.addEventListener("click", async () => {
                if (busy) return;
                setBusy(true);
                status.textContent = `Removing ${file.filename}...`;
                try {
                    const result = await responseData(await fetch("/documents", {
                        method: "DELETE", headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({filename: file.filename})
                    }));
                    status.textContent = result.message;
                    await loadDocuments();
                } catch (error) { status.textContent = error.message; }
                finally { setBusy(false); }
            });
            row.append(remove);
            list.append(row);
        }
    }

    form.addEventListener("submit", event => {
        event.preventDefault();
        if (busy) return;
        const files = selectedFiles();
        if (!files.length) { status.textContent = "Choose supported files or a folder first."; return; }
        if (files.length > limits.fileCount) { status.textContent = `Select at most ${limits.fileCount} documents per batch.`; return; }
        const oversized = files.find(file => file.size > limits.fileBytes);
        if (oversized) { status.textContent = `${oversized.name} exceeds the 500 MB per-file limit.`; return; }
        if (files.reduce((sum, file) => sum + file.size, 0) >= limits.batchBytes) {
            status.textContent = "Batch exceeds 1 GB. Select fewer files to leave room for request overhead.";
            return;
        }
        const body = new FormData();
        files.forEach(file => body.append("files", file, file.webkitRelativePath || file.name));
        setBusy(true);
        status.style.whiteSpace = "pre-wrap";
        status.textContent = "Uploading...";
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "/upload");
        xhr.upload.onprogress = event => {
            status.textContent = event.lengthComputable && event.loaded < event.total
                ? `Uploading: ${Math.round(event.loaded / event.total * 100)}%`
                : "Upload sent. Validating and indexing documents; this may take several minutes...";
        };
        xhr.onload = async () => {
            try {
                const data = await responseData(new Response(xhr.responseText, {status: xhr.status}));
                status.textContent = data.message;
                form.reset();
                updateSelection();
            } catch (error) { status.textContent = error.message; }
            try { await loadDocuments(); }
            catch (error) { status.textContent += ` ${error.message}`; }
            finally { setBusy(false); }
        };
        xhr.onerror = () => {
            status.textContent = "Connection lost. Refresh files to check whether the upload finished before retrying.";
            setBusy(false);
        };
        xhr.send(body);
    });
    refresh.addEventListener("click", () => loadDocuments().catch(error => { status.textContent = error.message; }));
    loadDocuments().catch(error => { status.textContent = error.message; });
})();
