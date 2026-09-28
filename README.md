# FRMC Copilot

The dashboard and MCP assessment tool now use:

`RAG -> Policy & Control -> Evidence -> Risk -> Review`

Uploaded documents are the first source for matching questions. Built-in SOX,
COSO and FRMC fundamentals remain available when no uploaded passage matches.
Agent assessments use uploaded controls and records, never the illustrative
500000 default. The latest uploaded file defining a given Control ID takes
priority; its timestamp is upload recency, not a verified policy effective date.
Removing a file removes it from subsequent assessments and answers. Reports
already saved retain their original assessment snapshots.

For an assessment, enter a Control ID (or mention it in the question) and a
Transaction ID. The existing control_register.csv is recognized by its actual
Control ID, Control Name, Control Description and Status columns. Assessable
controls must explicitly be Active. CSV and Excel rows retain filename, sheet
and row references. Other formats remain available as reference text.

Automatic policy interpretation is intentionally limited to the complete clause:
`All <transaction type> above <ISO currency> <amount> require approval from the preparer's designated finance manager before <posting|payment|execution>`.
This covers the existing FRMC-001 description without hard-coding its ID, currency
or threshold. Additional or unsupported conditions produce NOT_ASSESSED with
the actual requirement and source for human review; they are never silently ignored.

Upload transaction/approval evidence as CSV or Excel with these columns:

| Row type | Columns |
|---|---|
| Transaction | Record Type = transaction, Control ID, Transaction ID, Amount, Currency, Transaction Type, Designated Finance Manager, Posting At (or Payment At / Execution At) |
| Approval | Record Type = approval, Control ID, Transaction ID, Record ID, Amount, Currency, Status (Approved / Rejected / Pending), Reviewer, Recorded At |

Use timezone-bearing ISO timestamps, for example `2026-09-01T10:00:00Z`.
Transaction Type must match the type described in the control, e.g.
`manual journal entries`. Optional transaction columns are Entity ID and Criticality.
The approval must match transaction ID, amount, currency and designated reviewer,
and precede the required event. User-entered claims are compared with uploaded
records; entering Approved does not replace a missing approval file. Source
authenticity and the manager designation still require human review.

Risk ratings require an explicit uploaded CSV/Excel risk-rule table with columns:
`Record Type` (= risk_rule), `Control ID`, `Evidence Status`, `Criticality`,
`Risk Rating` (= Low / Medium / High). A rule must uniquely match the evidence
status and the **uploaded transaction's** criticality. No default risk matrix is
invented. Without a matching rule, risk is NOT_ASSESSED. Existing audit-finding
severities are returned separately under documented_risks; historical High
severity does not automatically rate the current transaction High.

`build_workflow(retriever)` without an uploaded context remains a legacy Python
compatibility entry point. The web/MCP flow passes UploadedKnowledge explicitly.
The original illustrative rule documentation below refers to that legacy mode.

Run with Python 3.10 or newer:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000 for the dashboard. Configure your existing .env using
.env.example as a reference. SOX/COSO fundamental answers are stored in
knowledge/fundamentals.json and work immediately without an API key or uploads.
They contain concise paraphrases with source URLs and review dates, rather
than complete policy texts. OPENROUTER_API_KEY is needed for uploading,
indexing, and searching additional documents. The embedding model uses the
OpenRouter embeddings endpoint.
Changing embedding models requires rebuilding a compatible vector index.

Uploads accept TXT, CSV, XLSX, XLS, PDF, and DOCX files in mixed batches of up to
1,000 files, 500 MiB per file, and 1 GiB total, including multipart overhead
(shown as MB/GB in the dashboard). Choose files and/or a folder; supported files
in nested folders are included, and unsupported folder files are skipped by the UI.
Folder paths are flattened into safe filenames, for example Finance/policy.pdf
becomes Finance_policy.pdf. Filename collisions reject the batch.
The extension is .pdf,
not .pdff. PDFs need extractable text (OCR is not included). Duplicate filenames are rejected.
Each file must pass a local finance/FRMC content screen before indexing. The
screen checks extracted text, not filenames, using corroborating finance,
SOX/COSO/FRMC, and control terminology. It is an English keyword heuristic:
ambiguous, short, or non-English financial documents can be rejected, and it
cannot guarantee perfect topic classification. Rules are in document_policy.py.
If any file fails validation, the entire batch is rejected with per-file reasons;
no files from that batch are indexed. Rebuild also validates existing documents.
Existing unrelated files are preserved on disk but block a rebuild until removed.
Previously indexed content is not automatically purged by this update.
Rebuild the knowledge base after manually changing files in uploads.

The Uploaded documents list below Upload shows stored filenames and sizes.
Remove deletes a file and its indexed passages immediately, including the last
uploaded file; built-in reference knowledge remains available. A removed filename
can be uploaded again. GET /documents lists uploads; DELETE /documents accepts
{"filename": "policy.pdf"}. If index deletion fails, the uploaded file is restored.
Upload progress is displayed while sending, followed by validation/indexing status.
Large files still require sufficient free disk, RAM, and processing time: parsers
extract full document text in memory. These limits are upload caps, not reserved
storage. Any reverse proxy must also allow a 1 GiB request and long processing times.

POST /query accepts {"question": "..."}. POST /run_agents additionally accepts
evidence (numeric amount and boolean approval), risk_factors (criticality and
boolean exception), and control_requirement. The current workflow retrieves
passages and applies simple evidence/risk rules. Reference questions return a
curated answer with sources; uploaded-document results are labeled as excerpts.
It does not generate an LLM answer or infer transaction evidence from question text.
POST /report stores title, content, and optional author.
GET /reports and GET /logs return saved records. Dashboard forms use these
routes and POST /ask, /build_rag, and /generate_report.

Run regression checks without calling a paid embedding service:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m pip check
```

The app uses a process-local lock for index updates; run one application process.

Try "What is SOX?", "What is Section 404?", "What are the five COSO components?",
or "Explain the 17 COSO principles" in the dashboard.

The agent workflow runs RAG -> Evidence -> Risk -> Review and returns agent_trace.
Use the dashboard's Run Evidence and Risk Agents form, or POST /run_agents:

```json
{
  "question": "What is SOX?",
  "evidence": {"amount": 600000, "approval": false},
  "risk_factors": {"criticality": "High"}
}
```

This example produces EXCEPTION and HIGH. An approved transaction produces PASS
and LOW when criticality is supplied. Missing evidence produces
INSUFFICIENT_EVIDENCE and NOT_ASSESSED, unless an explicit exception independently
supports a risk flag. These are illustrative rule results, not audit opinions.
Edit knowledge/control_rules.json to configure the approval threshold; 500000
is a project example, not a statutory SOX/COSO threshold. Free-text
control_requirement is retained for review, not automatically interpreted as a rule.

## Model Context Protocol (MCP)

mcp_server.py provides a real MCP server using the official Python SDK over stdio.
The former mcp.py logger is now audit_controller.py to avoid shadowing the SDK.
Flask and MCP share application routes, validation, reference memory, and database services.
The MCP adapter dispatches requests in-process; a running Flask web server is not required.

Available tools:

- ask_policy: answer a question with reference sources or uploaded document excerpts.
- assess_transaction: run the four agents and write an audit event.
- create_report: save a new report.
- list_reports: read report metadata.
- list_logs: read workflow audit events.

Resources: frmc://knowledge/fundamentals and frmc://knowledge/control-rules.

Copy the frmc server entry in mcp-client.example.json into an MCP client's
configuration that supports the mcpServers format, then restart/reconnect that
client. The example contains absolute paths for this workspace; update them if
you move the project. Other clients may require entering command and args in
their own configuration format. No client configuration has been modified automatically.
The server reads the existing project .env; no API key is needed for built-in questions.

The client starts this command itself:

```powershell
.\venv\Scripts\python.exe mcp_server.py
```

Running that command manually waits for MCP messages on stdin; it does not open
a web page. Stdio exposes no network listener. The dashboard remains available
by running app.py as before. The dashboard uses the shared application logic
directly; external MCP clients use the new protocol server.

Verify an actual client/server handshake, discovery, resource reads, all five tools,
and error recovery using an isolated temporary database:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_mcp_protocol.py -v
```

SDK reference: https://py.sdk.modelcontextprotocol.io/v1/
