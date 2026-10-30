// No dependencies, persisted tokens, document HTML, retries or implicit model calls.
const $ = (id) => document.getElementById(id);
let token = "", epoch = 0, offset = 0, historyOffset = 0;
let listRequest = 0, historyRequest = 0, sourceRequest = 0;
let regionView = null;
const maxPageOverlays = 512;
// Model request plus one input-page render and two citation renders, then transport margin.
const visualRequestTimeoutMs = 180000 + 3 * 10000 + 5000;
const pending = new Set();
const reasonText = {
  summary_scope_too_large: "This version exceeds the summary input limit. Nothing was silently shortened.",
  summary_source_warnings: "The extracted text has warnings. A complete summary input cannot be confirmed.",
  summary_source_incomplete: "The complete source could not be verified. No highlights were created.",
  summary_empty_source: "This version has no text to summarize.",
  summary_page_ambiguous: "Precise page citations are unavailable for this version.",
  summary_token_budget_unavailable: "The model summary token budget is not configured.",
  summary_model_binding_unavailable: "The selected model, tokenizer and prompt budget have not been verified.",
  summary_token_budget_exceeded: "This version does not fit the configured model context. Nothing was silently shortened.",
  summary_token_counter_invalid: "The configured tokenizer could not verify the complete prompt budget.",
  summary_no_extractable_points: "No bounded extracts could be selected from this text.",
  no_authorized_evidence: "No matching source is available to your account.",
  no_visual_evidence: "No authorized PDF page matched this visual question.",
  insufficient_visual_support: "The page does not support a reliable visual answer.",
  visual_model_not_configured: "No unique table cell matched this question. A visual model is not configured.",
  insufficient_support: "The available sources do not support an answer.",
  model_abstained: "The model could not support an answer from these sources.",
  conflicting_evidence: "The sources disagree. An answer would be unreliable.",
  evidence_no_longer_authorized: "The source changed or access was removed. Search again.",
  unsupported_model_output: "The proposed answer could not be verified against its source.",
};
const errorText = {
  summary_not_configured: "The selected summary service is not configured.",
  model_not_configured: "An answer model has not been configured. You can still find passages.",
  semantic_answer_not_configured: "The semantic service is not configured.",
  model_unavailable: "The configured model is unavailable. Try again later.",
  semantic_answer_unavailable: "The semantic service is unavailable. Try again later.",
  visual_answer_not_configured: "The local visual model or page renderer is not configured.",
  visual_answer_unavailable: "The visual answer could not be completed.",
  visual_answer_invalid_response: "The visual model returned an answer that could not be verified.",
  visual_evidence_not_found: "The source page changed or is no longer available to your account.",
  query_history_not_found: "This saved result is no longer available to your account.",
  upload_too_large: "This document exceeds the upload limit.",
  upload_forbidden: "You do not have permission to upload this document.",
  region_unavailable: "The page preview is unavailable right now.",
};
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function reset() {
  epoch += 1; listRequest += 1; historyRequest += 1; sourceRequest += 1;
  for (const controller of pending) controller.abort();
  pending.clear(); token = ""; offset = 0; historyOffset = 0;
  $("token").value = ""; $("question").value = ""; $("file").value = "";
  $("documents").replaceChildren(); $("answer-area").replaceChildren();
  $("history-items").replaceChildren(); $("history-empty").hidden = true;
  $("more-history").hidden = true; $("history-section").open = false;
  $("source-text").replaceChildren(); $("source-meta").textContent = "";
  $("source-text").hidden = false; $("source-table").hidden = true;
  $("source-table-content").replaceChildren();
  $("source-dialog").close(); $("source-title").textContent = "Source";
  regionView = null; $("source-view").hidden = true;
  summaryRequest += 1; answerRequest += 1; selectedSummary = null; summaryResultRequest = null;
  $("summary-dialog").close(); $("summary-version").replaceChildren();
  $("summary-mode").value = "extractive"; $("create-summary").textContent = "Create highlights";
  $("summary-title").textContent = "Document highlights";
  $("upload-status").textContent = ""; $("notice").textContent = "";
  $("connection").hidden = false; $("workspace").hidden = true;
  $("disconnect").hidden = true; $("more").hidden = true;
  document.querySelectorAll("button").forEach((button) => { button.disabled = false; });
  $("file").disabled = false; $("mode").disabled = false;
  $("mode").value = "sources";
  $("source-query").value = ""; $("source-query-field").hidden = true;
  $("mode-caption").textContent = "Find passages · no model";
  $("ask").textContent = "Find";
}
async function api(path, options = {}, binary = false) {
  const generation = epoch, controller = new AbortController();
  pending.add(controller);
  const timer = setTimeout(() => controller.abort(), path === "/v1/visual-answer" ? visualRequestTimeoutMs : 95000);
  try {
    const response = await fetch(path, {
      ...options, signal: controller.signal, cache: "no-store", credentials: "omit",
      headers: { ...options.headers, Authorization: `Bearer ${token}` },
    });
    if (generation !== epoch) throw new DOMException("Session changed", "AbortError");
    if (response.status === 401) {
      reset(); throw new Error("Access token expired or was rejected. Connect again.");
    }
    if (binary && response.ok && response.headers.get("Content-Type") !== "image/png") throw new Error("The source image could not be verified.");
    const data = binary && response.ok ? await response.blob() : await response.json();
    if (generation !== epoch) throw new DOMException("Session changed", "AbortError");
    if (!response.ok) {
      const code = data.detail?.code;
      const error = new Error(errorText[code] || (response.status === 404
        ? "The source is no longer available to your account."
        : `The request could not be completed (${response.status}).`));
      error.status = response.status; error.code = code;
      throw error;
    }
    return data;
  } finally { clearTimeout(timer); pending.delete(controller); }
}
function notice(error) {
  if (error.name !== "AbortError") $("notice").textContent = error.message;
}
async function documents(append = false) {
  const request = ++listRequest;
  const next = append ? offset : 0;
  const data = await api(`/v1/documents?limit=20&offset=${next}`);
  if (request !== listRequest) return;
  if (!append) $("documents").replaceChildren();
  for (const item of data.items) {
    const row = node("li", item.title);
    const warningCodes = Array.isArray(item.ingestion_warnings) ? item.ingestion_warnings : [];
    const ocrPages = warningCodes.map((warning) => typeof warning === "string"
      ? /^ocr_required_page_(\d+)$/.exec(warning)?.[1] : null).filter(Boolean);
    const warningText = ocrPages.length
      ? ` · OCR needed on page${ocrPages.length === 1 ? "" : "s"} ${ocrPages.join(", ")}`
      : warningCodes.length ? " · Extraction warning" : "";
    row.append(node("span", `${item.status.replaceAll("_", " ")}${warningText}`,
      warningCodes.length ? "document-status document-warning" : "document-status"));
    if (item.active_version_id) {
      const button = node("button", "Highlights", "quiet");
      button.addEventListener("click", () => guarded(button, () => chooseSummary(item)));
      row.append(button);
    }
    $("documents").append(row);
  }
  if (!data.items.length && !append) $("documents").append(node("li", "No documents yet.", "hint"));
  offset = next + data.items.length;
  $("more").hidden = data.items.length < 20;
}
function historyStatus(status) {
  return ({
    answered: "Answered", abstained: "No supported answer",
    evidence_only: "Sources found", failed: "Failed", running: "In progress",
  })[status] || "Saved result";
}
async function recentQuestions(append = false) {
  const request = ++historyRequest;
  const next = append ? historyOffset : 0;
  const data = await api(`/v1/queries?limit=10&offset=${next}`);
  if (request !== historyRequest) return;
  if (!append) $("history-items").replaceChildren();
  for (const item of data.items) {
    if (!item || !/^[a-f0-9]{32}$/.test(item.query_run_id)
        || typeof item.query_text !== "string") continue;
    const row = node("li", undefined, "history-item");
    const open = node("button", item.query_text, "quiet history-question");
    open.type = "button";
    open.addEventListener("click", () => guarded(open, () => openSavedQuery(item.query_run_id)));
    const elapsed = Number.isFinite(item.elapsed_seconds)
      ? ` · ${item.elapsed_seconds.toFixed(1)}s` : "";
    row.append(open, node("span", `${historyStatus(item.status)}${elapsed}`, "document-status"));
    $("history-items").append(row);
  }
  historyOffset = Number.isInteger(data.next_offset) ? data.next_offset : next;
  $("history-empty").hidden = $("history-items").children.length > 0 || append;
  $("more-history").hidden = data.next_offset === null;
}
function renderSavedQuery(record) {
  $("answer-area").replaceChildren(node("h2", "Saved result"));
  $("answer-area").append(node("p", record.query_text, "muted saved-question"));
  const elapsed = Number.isFinite(record.stage_latencies?.total_seconds)
    ? ` · ${record.stage_latencies.total_seconds.toFixed(1)}s` : "";
  $("answer-area").append(node("p", `${historyStatus(record.status)}${elapsed}`, "hint"));
  const timingLabels = {
    retrieval: "Retrieval",
    context_packing: "Evidence selection",
    table_lookup: "Table lookup",
    native_chart_lookup: "Chart lookup",
    parent_expansion: "Parent context",
    visual_render: "Page rendering",
    authorization_recheck: "Access recheck",
    answer_model: "Answer model",
    citation_validation: "Citation check",
    total_seconds: "Total",
  };
  const timings = record.stage_latencies;
  if (timings && typeof timings === "object") {
    const trace = node("details", undefined, "query-trace");
    trace.append(node("summary", "Trace"));
    const path = node("p", `Path · ${record.retrieval_strategy}`, "hint");
    trace.append(path);
    const steps = node("ul", undefined, "trace-list");
    for (const [key, label] of Object.entries(timingLabels)) {
      if (!Number.isFinite(timings[key]) || timings[key] < 0) continue;
      steps.append(node("li", `${label} · ${timings[key].toFixed(3)}s`));
    }
    if (steps.children.length) trace.append(steps);
    $("answer-area").append(trace);
  }
  if (record.result) renderAnswer(record.result, record.id);
}
async function openSavedQuery(queryId) {
  const request = ++answerRequest, generation = epoch;
  $("answer-area").replaceChildren(node("p", "Opening saved result…", "muted"));
  const record = await api(`/v1/queries/${encodeURIComponent(queryId)}`);
  if (request !== answerRequest || generation !== epoch) return;
  if (record.id !== queryId) throw new Error("The saved result could not be verified.");
  renderSavedQuery(record);
}
async function guarded(button, action) {
  if (button.disabled) return;
  const generation = epoch;
  button.disabled = true; $("notice").textContent = "";
  try { await action(); } catch (error) { notice(error); }
  finally { if (generation === epoch) button.disabled = false; }
}
$("connect-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = $("token").value.trim();
  reset(); token = value;
  await guarded(event.submitter, async () => {
    try { await documents(); await recentQuestions(); }
    catch (error) { reset(); throw error; }
    $("connection").hidden = true; $("workspace").hidden = false;
    $("disconnect").hidden = false; $("question").focus();
  });
});
$("disconnect").addEventListener("click", () => { reset(); $("token").focus(); });
$("refresh").addEventListener("click", () => guarded($("refresh"), async () => {
  await documents(); await recentQuestions();
}));
$("more").addEventListener("click", () => guarded($("more"), () => documents(true)));
$("more-history").addEventListener("click", () => guarded($("more-history"), () => recentQuestions(true)));
$("file").addEventListener("change", () => guarded($("file"), async () => {
  const file = $("file").files[0];
  if (!file) return;
  if (!/\.(txt|md|pdf)$/i.test(file.name) || !file.size) throw new Error("Choose a nonempty TXT, Markdown or native PDF file.");
  const encoded = encodeURIComponent(file.name);
  const result = await api("/v1/documents", { method: "POST", body: file, headers: {
    "Content-Type": "application/octet-stream", "X-Canonical-Key": `web/${encoded}`,
    "X-Title*": `UTF-8''${encoded}`, "X-Filename*": `UTF-8''${encoded}`,
  }});
  $("upload-status").textContent = result.duplicate ? "Already uploaded. Refresh to see its status." : "Upload accepted. It will appear after processing and access is granted. Refresh to check.";
  $("file").value = "";
  await documents();
}));
function sameSource(item, live) {
  return item.document_id === live.document_id && item.document_version_id === live.document_version_id
    && item.run_id === live.run_id;
}
function showTableCells(cells) {
  if (!Array.isArray(cells) || cells.length === 0) return false;
  const rows = new Map();
  for (const cell of cells) {
    if (!Number.isInteger(cell.row_index) || !Number.isInteger(cell.column_index)
        || cell.row_index < 0 || cell.column_index < 0 || typeof cell.text !== "string") continue;
    if (!rows.has(cell.row_index)) rows.set(cell.row_index, new Map());
    rows.get(cell.row_index).set(cell.column_index, cell.text);
  }
  if (rows.size === 0) return false;
  const table = node("table", undefined, "source-table-grid");
  table.setAttribute("aria-label", "Extracted table row");
  for (const [rowIndex, columns] of [...rows].sort((a, b) => a[0] - b[0])) {
    const row = node("tr");
    const maxColumn = Math.max(...columns.keys());
    for (let columnIndex = 0; columnIndex <= maxColumn; columnIndex += 1) {
      row.append(node("td", columns.get(columnIndex) ?? ""));
    }
    table.append(row);
  }
  $("source-table-content").replaceChildren(table);
  $("source-table").hidden = false;
  return true;
}
async function openSource(item) {
  const request = ++sourceRequest, answerGeneration = answerRequest;
  regionView = null; $("source-view").hidden = true;
  $("source-text").replaceChildren(); $("source-meta").textContent = "";
  $("source-text").hidden = false; $("source-table").hidden = true;
  $("source-table-content").replaceChildren();
  $("source-dialog").close();
  const live = await api(`/v1/evidence/${encodeURIComponent(item.evidence_id || item.chunk_id)}`);
  if (request !== sourceRequest || answerGeneration !== answerRequest) return;
  if (!sameSource(item, live)) throw new Error("The source changed. Search again for the current version.");
  const cellPayload = await api(`/v1/evidence/${encodeURIComponent(item.evidence_id || item.chunk_id)}/table-cells`);
  if (request !== sourceRequest || answerGeneration !== answerRequest) return;
  $("source-title").textContent = live.title;
  $("source-meta").textContent = live.page_numbers?.length ? `Page ${live.page_numbers.join(", ")}` : live.filename;
  const start = item.chunk_start_char, end = item.chunk_end_char;
  // Server offsets are Unicode code points, not JavaScript UTF-16 indices.
  const chars = Array.from(live.text);
  if (Number.isInteger(start) && Number.isInteger(end) && start >= 0 && end > start
      && end <= chars.length && chars.slice(start, end).join("") === item.quote) {
    $("source-text").append(document.createTextNode(chars.slice(0, start).join("")),
      node("mark", item.quote), document.createTextNode(chars.slice(end).join("")));
  } else $("source-text").textContent = live.text;
  if (showTableCells(cellPayload.cells)) $("source-text").hidden = true;
  $("source-dialog").showModal();
}
async function openRegion(item, region, pageRegion, overlays = [], overlayPageBinding = null, selectedArea = false) {
  const request = ++sourceRequest, answerGeneration = answerRequest;
  regionView = null; $("source-view").hidden = true;
  $("source-text").replaceChildren(); $("source-meta").textContent = "";
  $("source-text").hidden = false; $("source-table").hidden = true;
  $("source-table-content").replaceChildren();
  $("source-dialog").close();
  if (!/^[a-f0-9]{32}$/.test(region.id)) throw new Error("The source image could not be verified.");
  const liveRegion = await api(`/v1/regions/${region.id}`);
  if (request !== sourceRequest || answerGeneration !== answerRequest) return;
  if (!sameSource(item, liveRegion) || !item.page_numbers?.includes(liveRegion.page_number)) throw new Error("The source changed. Open the saved result again.");
  let live = liveRegion;
  let bbox = null;
  if (pageRegion) {
    if (liveRegion.kind !== "explicitcrop" || !/^[a-f0-9]{32}$/.test(pageRegion.id)) {
      throw new Error("The source image could not be verified.");
    }
    live = await api(`/v1/regions/${pageRegion.id}`);
    if (request !== sourceRequest || answerGeneration !== answerRequest) return;
    if (!sameSource(item, live) || !item.page_numbers?.includes(live.page_number)
        || live.kind !== "wholepage" || live.query_run_id !== liveRegion.query_run_id
        || live.citation_index !== liveRegion.citation_index
        || live.page_number !== liveRegion.page_number
        || live.page_image_sha256 !== liveRegion.page_image_sha256
        || live.page_width !== liveRegion.page_width || live.page_height !== liveRegion.page_height
        || live.applied_rotation_degrees !== liveRegion.applied_rotation_degrees) {
      throw new Error("The source image could not be verified.");
    }
    bbox = liveRegion.normalized_bbox;
    if (!validPageBox(bbox)) {
      throw new Error("The source image could not be verified.");
    }
    if (selectedArea) { live = liveRegion; bbox = null; }
  }
  if (overlayPageBinding && (
    overlayPageBinding.source_pdf_sha256 !== live.source_pdf_sha256
      || overlayPageBinding.page_image_sha256 !== live.page_image_sha256
      || overlayPageBinding.page_width !== live.page_width
      || overlayPageBinding.page_height !== live.page_height
      || overlayPageBinding.applied_rotation_degrees !== live.applied_rotation_degrees
  )) {
    throw new Error("The source image could not be verified.");
  }
  const boxes = bbox ? [{ bbox }] : [];
  if (!Array.isArray(overlays) || overlays.length > maxPageOverlays
      || overlays.some((overlay) => !validPageBox(overlay?.bbox))) {
    throw new Error("The source image could not be verified.");
  }
  if (!selectedArea) boxes.push(...overlays);
  const blob = await api(`/v1/regions/${live.id}/image`, {}, true);
  if (request !== sourceRequest || answerGeneration !== answerRequest) return;
  const generation = epoch, bitmap = await createImageBitmap(blob);
  try {
    if (generation !== epoch || request !== sourceRequest || answerGeneration !== answerRequest) return;
    if (bitmap.width !== live.width || bitmap.height !== live.height) throw new Error("The source image could not be verified.");
    const canvas = node("canvas");
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `${item.title}, page ${live.page_number}${selectedArea ? ", selected area" : boxes.length ? `, ${boxes.length} cited areas highlighted` : ""}`);
    const context = canvas.getContext("2d");
    if (!context) throw new Error("The source image could not be displayed.");
    context.drawImage(bitmap, 0, 0);
    for (const overlay of boxes) {
      const [x0, y0, x1, y1] = overlay.bbox;
      const x = x0 * canvas.width, y = y0 * canvas.height;
      const width = (x1 - x0) * canvas.width, height = (y1 - y0) * canvas.height;
      context.fillStyle = "rgba(224, 183, 77, .18)";
      context.strokeStyle = "#9a4f32";
      context.lineWidth = Math.max(2, Math.min(canvas.width, canvas.height) * .0025);
      context.fillRect(x, y, width, height);
      context.strokeRect(x, y, width, height);
    }
    $("source-title").textContent = item.title;
    $("source-meta").textContent = `Page ${live.page_number}${boxes.length ? ` · ${boxes.length} highlighted areas` : live.kind === "wholepage" ? " · Full page" : " · Selected area"}`;
    if (pageRegion) {
      regionView = {item, region, pageRegion, overlays, overlayPageBinding, selectedArea};
      $("source-view").textContent = selectedArea ? "Full page" : "Selected area";
      $("source-view").hidden = false;
    }
    $("source-text").replaceChildren(canvas); $("source-dialog").showModal();
  } finally { bitmap.close(); }
}
$("source-view").addEventListener("click", () => guarded($("source-view"), async () => {
  const view = regionView;
  if (!view) return;
  await openRegion(view.item, view.region, view.pageRegion, view.overlays,
    view.overlayPageBinding, !view.selectedArea);
}));
function validPageBox(box) {
  return Array.isArray(box) && box.length === 4
    && box.every((value) => typeof value === "number" && Number.isFinite(value))
    && 0 <= box[0] && box[0] < box[2] && box[2] <= 1
    && 0 <= box[1] && box[1] < box[3] && box[3] <= 1;
}
async function openCitationPage(item, queryId, citationIndex) {
  const request = ++sourceRequest, generation = epoch, answerGeneration = answerRequest;
  regionView = null; $("source-view").hidden = true;
  const regions = Array.isArray(item.visual_regions) ? item.visual_regions : [];
  const cropRegion = regions.find((candidate) => candidate.kind === "explicitcrop");
  let pageRegion = regions.find((candidate) => candidate.kind === "wholepage");
  const evidenceId = item.evidence_id || item.chunk_id;
  const tablePayload = await api(`/v1/evidence/${encodeURIComponent(evidenceId)}/table-cells`);
  if (request !== sourceRequest || generation !== epoch || answerGeneration !== answerRequest) return;
  if (!tablePayload || typeof tablePayload !== "object"
      || tablePayload.chunk_id !== evidenceId || !Array.isArray(tablePayload.cells)) {
    throw new Error("The source image could not be verified.");
  }
  if (tablePayload.cells.length > maxPageOverlays) {
    throw new Error("This citation has too many regions to preview at once.");
  }
  const overlays = [];
  for (const cell of tablePayload.cells) {
    if (!cell || typeof cell !== "object" || cell.page_number !== item.page_numbers?.[0]) {
      throw new Error("The source image could not be verified.");
    }
    overlays.push({ bbox: cell.bbox });
  }
  let overlayPageBinding = null;
  try {
    const ocrPayload = await api(`/v1/queries/${encodeURIComponent(queryId)}/citations/${citationIndex}/ocr-regions`);
    if (request !== sourceRequest || generation !== epoch || answerGeneration !== answerRequest) return;
    if (!ocrPayload || typeof ocrPayload !== "object") {
      throw new Error("The source image could not be verified.");
    }
    const quoteSpan = ocrPayload.quote_span;
    if (ocrPayload.query_run_id !== queryId || ocrPayload.citation_index !== citationIndex
        || ocrPayload.evidence_id !== evidenceId || ocrPayload.page_number !== item.page_numbers?.[0]
        || ocrPayload.document_id !== item.document_id
        || ocrPayload.document_version_id !== item.document_version_id
        || ocrPayload.run_id !== item.run_id
        || (typeof item.quote === "string" && ocrPayload.quote !== item.quote)
        || !/^[a-f0-9]{64}$/.test(ocrPayload.source_pdf_sha256 || "")
        || !/^[a-f0-9]{64}$/.test(ocrPayload.page_image_sha256 || "")
        || !Number.isInteger(ocrPayload.page_width) || ocrPayload.page_width < 1
        || !Number.isInteger(ocrPayload.page_height) || ocrPayload.page_height < 1
        || ![0, 90, 180, 270].includes(ocrPayload.applied_rotation_degrees)
        || !quoteSpan || quoteSpan.offset_unit !== "unicode_code_points"
        || !Number.isInteger(quoteSpan.start_char) || quoteSpan.start_char < 0
        || !Number.isInteger(quoteSpan.end_char) || quoteSpan.end_char <= quoteSpan.start_char
        || !Array.isArray(ocrPayload.regions)) {
      throw new Error("The source image could not be verified.");
    }
    if (ocrPayload.regions.length > maxPageOverlays - overlays.length) {
      throw new Error("This citation has too many regions to preview at once.");
    }
    overlayPageBinding = {
      source_pdf_sha256: ocrPayload.source_pdf_sha256,
      page_image_sha256: ocrPayload.page_image_sha256,
      page_width: ocrPayload.page_width,
      page_height: ocrPayload.page_height,
      applied_rotation_degrees: ocrPayload.applied_rotation_degrees,
    };
    for (const region of ocrPayload.regions) {
      if (!region || typeof region !== "object"
          || !Number.isInteger(region.start_char) || region.start_char < 0
          || !Number.isInteger(region.end_char) || region.end_char <= region.start_char
          || region.start_char >= quoteSpan.end_char || region.end_char <= quoteSpan.start_char
          || !validPageBox(region.bbox)) {
        throw new Error("The source image could not be verified.");
      }
      overlays.push({ bbox: region.bbox });
    }
  } catch (error) {
    if (!(error.status === 404 && error.code === "region_not_found")) throw error;
  }
  if (request !== sourceRequest || generation !== epoch || answerGeneration !== answerRequest) return;
  if (overlays.length > maxPageOverlays || overlays.some((overlay) => !validPageBox(overlay.bbox))) {
    throw new Error("The cited area has too many parts to highlight here.");
  }
  if (!pageRegion) {
    pageRegion = await api(`/v1/queries/${encodeURIComponent(queryId)}/citations/${citationIndex}/regions`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
    });
  }
  if (request !== sourceRequest || generation !== epoch || answerGeneration !== answerRequest) return;
  await openRegion(
    item,
    cropRegion || pageRegion,
    cropRegion ? pageRegion : undefined,
    overlays,
    overlayPageBinding,
  );
}
function sources(items, queryId) {
  if (!items.length) { $("answer-area").append(node("p", "No matching sources found.", "muted")); return; }
  $("answer-area").append(node("h2", "Sources"));
  for (const [index, item] of items.entries()) {
    const article = node("article", undefined, "source-item");
    const pages = item.page_numbers?.length ? ` · p. ${item.page_numbers.join(", ")}` : "";
    const link = node("button", item.title + pages, "source-link quiet");
    link.addEventListener("click", () => guarded(link, () => openSource(item)));
    const preview = Array.from(item.quote ?? item.text);
    article.append(link, node("p", preview.slice(0, 420).join("") + (preview.length > 420 ? "…" : ""), "source-quote"));
    const regions = Array.isArray(item.visual_regions) ? item.visual_regions : [];
    const pageRegion = regions.find((candidate) => candidate.kind === "wholepage");
    const cropRegion = regions.find((candidate) => candidate.kind === "explicitcrop");
    const region = pageRegion || cropRegion || regions[0];
    if (queryId && typeof item.filename === "string" && item.filename.toLowerCase().endsWith(".pdf")
        && item.page_numbers?.length === 1) {
      const image = node("button", "View on page", "quiet");
      image.addEventListener("click", () => guarded(image, () => openCitationPage(
        item, queryId, item.citation_index ?? index + 1,
      )));
      article.append(image);
    } else if (region) {
      const withOverlay = pageRegion && cropRegion;
      const image = node("button", withOverlay ? "View on page" : region.kind === "wholepage" ? "Page image" : "Selected area", "quiet");
      image.addEventListener("click", () => guarded(image, () => openRegion(
        item,
        withOverlay ? cropRegion : region,
        withOverlay ? pageRegion : undefined,
      )));
      article.append(image);
    }
    $("answer-area").append(article);
  }
}
$("mode").addEventListener("change", () => {
  $("source-query-field").hidden = $("mode").value !== "visual";
  $("mode-caption").textContent = $("mode").selectedOptions[0].textContent;
  $("ask").textContent = $("mode").value.endsWith("sources") ? "Find" : "Ask";
  $("answer-area").replaceChildren();
});
$("ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  guarded($("ask"), async () => {
    const generation = epoch;
    const request = ++answerRequest;
    const question = $("question").value.trim(), mode = $("mode").value;
    if (!question) return;
    $("answer-area").replaceChildren(node("p", mode.endsWith("sources") ? "Finding passages…" : "Checking the sources…", "muted"));
    $("mode").disabled = true;
    try {
      let result;
      if (mode === "sources") {
        result = await api(`/v1/search?q=${encodeURIComponent(question)}&limit=4`);
        if (request !== answerRequest) return;
        $("answer-area").replaceChildren(); sources(result.items); return;
      }
      const semantic = mode.startsWith("semantic") || mode === "parent";
      const visual = mode === "visual";
      const sourceQuery = visual ? $("source-query").value.trim() : "";
      result = await api(visual ? "/v1/visual-answer" : semantic ? "/v1/semantic-answer" : "/v1/answer", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(semantic ? { question, mode: mode === "semantic-sources" ? "evidence_only" : "extractive", strategy: mode === "parent" ? "lexical" : "hybrid" } : sourceQuery ? { question, search_query: sourceQuery } : { question }),
      });
      if (request !== answerRequest) return;
      $("answer-area").replaceChildren();
      if (mode === "semantic-sources") {
        sources(result.retrieval.items.map((hit) => hit.evidence), result.query_run_id);
        return;
      }
      const answer = semantic || visual ? result.result : result;
      const queryId = result.query_run_id || answer.query_run_id || result.id;
      renderAnswer(answer, queryId);
      if (queryId) {
        try { await recentQuestions(); } catch (error) { notice(error); }
        const saved = node("button", "Saved result", "quiet");
        saved.type = "button";
        saved.addEventListener("click", () => guarded(saved, () => openSavedQuery(queryId)));
        $("answer-area").append(saved);
      }
    } catch (error) {
      if (generation === epoch && request === answerRequest) $("answer-area").replaceChildren();
      throw error;
    } finally { if (generation === epoch) $("mode").disabled = false; }
  });
});
function renderAnswer(answer, queryId) {
  if (answer.abstained) $("answer-area").append(node("p", reasonText[answer.abstention_reason] || "An answer could not be verified from these sources.", "muted"));
  else { $("answer-area").append(node("p", answer.answer, "answer-text")); sources(answer.citations, queryId); }
}
let summaryRequest = 0, answerRequest = 0, selectedSummary = null, summaryResultRequest = null;
function cancelSummary() {
  if (summaryResultRequest === answerRequest) {
    answerRequest += 1; $("answer-area").replaceChildren();
  }
  summaryResultRequest = null; summaryRequest += 1; selectedSummary = null;
  $("summary-dialog").close();
}
async function chooseSummary(item) {
  cancelSummary();
  const request = ++summaryRequest;
  selectedSummary = null; $("summary-dialog").close(); $("summary-version").replaceChildren();
  const versions = await api(`/v1/documents/${encodeURIComponent(item.id)}/versions`);
  if (request !== summaryRequest) return;
  const active = versions.items.filter((version) => version.id === item.active_version_id && version.status === "active");
  if (active.length !== 1) throw new Error("The current version changed. Refresh the document list.");
  selectedSummary = {id: item.id, title: item.title, version: active[0].id};
  $("summary-title").textContent = item.title;
  const option = node("option", `Version ${active[0].version_number} · current`);
  option.value = active[0].id;
  $("summary-version").append(option); $("summary-version").value = option.value;
  $("summary-dialog").showModal();
}
$("summary-form").addEventListener("submit", (event) => {
  event.preventDefault();
  guarded($("create-summary"), async () => {
    const selected = selectedSummary, scopeRequest = summaryRequest;
    if (!selected || $("summary-version").value !== selected.version) return;
    const mode = $("summary-mode").value;
    if (!["model", "extractive"].includes(mode)) return;
    const request = ++answerRequest;
    summaryResultRequest = request;
    $("answer-area").replaceChildren(node("p", mode === "model" ? "Preparing a supported summary…" : "Selecting document highlights…", "muted"));
    try {
      const result = await api(`/v1/documents/${encodeURIComponent(selected.id)}/summary`, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({version_id: selected.version, mode}),
      });
      if (request !== answerRequest || scopeRequest !== summaryRequest) return;
      $("summary-dialog").close();
      $("answer-area").replaceChildren(node("h2", `${selected.title} · ${mode === "model" ? "summary" : "highlights"}`));
      if (result.abstained) $("answer-area").append(node("p", reasonText[result.abstention_reason] || "Highlights could not be verified.", "muted"));
      else if (mode === "model") {
        $("answer-area").append(node("p", "Model-generated summary. Check the cited passages; meaning and completeness are not independently verified.", "hint"));
        for (const paragraph of result.paragraphs) {
          const text = node("p", paragraph.text, "answer-text");
          for (const index of paragraph.citation_indexes) {
            const citation = result.citations[index - 1];
            if (!citation) throw new Error("The summary citation is incomplete.");
            const link = node("button", `[${index}]`, "quiet source-link");
            link.addEventListener("click", () => guarded(link, () => openSource(citation)));
            text.append(link);
          }
          $("answer-area").append(text);
        }
      }
      else {
        $("answer-area").append(node("p", "All canonical text considered. Selected extracts; details may be omitted.", "hint"));
        sources(result.citations, result.query_run_id);
      }
    } catch (error) {
      if (request === answerRequest) $("answer-area").replaceChildren();
      throw error;
    } finally { if (summaryResultRequest === request) summaryResultRequest = null; }
  });
});
$("close-summary").addEventListener("click", cancelSummary);
$("summary-dialog").addEventListener("cancel", cancelSummary);
$("summary-mode").addEventListener("change", () => { $("create-summary").textContent = $("summary-mode").value === "model" ? "Create summary" : "Create highlights"; });
$("close-source").addEventListener("click", () => {
  regionView = null; $("source-view").hidden = true;
  sourceRequest += 1; $("source-dialog").close(); $("source-text").replaceChildren();
  $("source-text").hidden = false; $("source-table").hidden = true;
  $("source-table-content").replaceChildren();
});
$("source-dialog").addEventListener("cancel", () => {
  regionView = null; $("source-view").hidden = true;
  sourceRequest += 1; $("source-text").replaceChildren();
  $("source-text").hidden = false; $("source-table").hidden = true;
  $("source-table-content").replaceChildren();
});
window.addEventListener("pagehide", reset);
