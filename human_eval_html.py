import argparse
import json
import os
import time


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description="Build a local HTML page for blind A/B human evaluation."
    )
    parser.add_argument(
        "--manifest", default="data/output/supplementary/human_eval/manifest.jsonl"
    )
    parser.add_argument(
        "--out", default="data/output/supplementary/human_eval/index.html"
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Build a page for only this many manifest items, starting at start_index.",
    )
    parser.add_argument(
        "--storage_key",
        default=None,
        help="Browser localStorage key. A separate key is chosen automatically for a subset.",
    )
    parser.add_argument(
        "--download_name",
        default=None,
        help="Filename used by the Export button.",
    )
    args = parser.parse_args()

    if args.start_index < 0 or (args.max_items is not None and args.max_items <= 0):
        parser.error("start_index must be nonnegative and max_items must be positive")

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    all_items = list(read_jsonl(args.manifest))
    stop = None if args.max_items is None else args.start_index + args.max_items
    selected_items = all_items[args.start_index:stop]
    items = []
    for offset, item in enumerate(selected_items):
        panel = item["panel"]
        item = dict(item)
        rel_panel = os.path.relpath(panel, out_dir)
        cache_key = int(os.path.getmtime(panel)) if os.path.exists(panel) else int(time.time())
        item["panel"] = f"{rel_panel}?v={cache_key}"
        item["display_index"] = args.start_index + offset + 1
        items.append(item)

    if not items:
        parser.error("the requested manifest slice contains no items")

    subset_end = args.start_index + len(items)
    storage_key = args.storage_key
    if storage_key is None:
        if args.start_index == 0 and args.max_items is None:
            storage_key = "t2t_vicl_human_eval_responses"
        else:
            storage_key = f"t2t_vicl_human_eval_responses_{args.start_index}_{subset_end}"
    download_name = args.download_name or (
        "responses.jsonl"
        if args.start_index == 0 and args.max_items is None
        else f"responses_{args.start_index:04d}_{subset_end - 1:04d}.jsonl"
    )

    has_reference = any(item.get("include_reference") for item in items)
    hint_text = (
        "For each item, use the reference output as guidance and choose which anonymized output better matches the target transformation for the query image while preserving relevant image content. Use Tie if neither side is clearly better."
        if has_reference
        else "For each item, choose which output better performs the target transformation for the query image while preserving relevant image content. Use Tie if neither side is clearly better."
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>T2T-VICL Human Evaluation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; }}
    header {{ position: sticky; top: 0; background: white; padding: 12px 0 16px; border-bottom: 1px solid #d8dee6; z-index: 10; }}
    h1 {{ margin: 0 0 6px; font-size: 22px; }}
    .hint {{ color: #52606d; max-width: 980px; line-height: 1.45; }}
    .toolbar {{ display: flex; gap: 12px; align-items: center; margin-top: 12px; flex-wrap: wrap; }}
    button {{ border: 1px solid #9aa5b1; background: #f5f7fa; padding: 8px 12px; border-radius: 6px; cursor: pointer; }}
    button:hover {{ background: #e4e7eb; }}
    .progress {{ font-weight: 600; }}
    .item {{ padding: 22px 0; border-bottom: 1px solid #e4e7eb; }}
    .meta {{ color: #52606d; margin-bottom: 8px; font-size: 14px; }}
    img {{ max-width: min(100%, 820px); border: 1px solid #d8dee6; display: block; }}
    .choices {{ display: flex; gap: 10px; margin-top: 10px; flex-wrap: wrap; }}
    .choice {{ border: 1px solid #bcccdc; border-radius: 6px; padding: 8px 12px; cursor: pointer; user-select: none; }}
    .choice input {{ margin-right: 6px; }}
    .choice:has(input:checked) {{ border-color: #2563eb; background: #dbeafe; }}
    textarea {{ width: min(100%, 980px); height: 180px; margin-top: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  </style>
</head>
<body>
  <header>
    <h1>T2T-VICL Human Evaluation</h1>
    <div class="hint">
      {hint_text}
    </div>
    <div class="toolbar">
      <button onclick="exportJsonl()">Export responses.jsonl</button>
      <button onclick="copyJsonl()">Copy JSONL to clipboard</button>
      <button onclick="clearResponses()">Clear annotations</button>
      <span class="progress" id="progress"></span>
    </div>
    <textarea id="jsonl" placeholder="Exported JSONL will appear here."></textarea>
  </header>
  <main id="items"></main>

  <script>
    const ITEMS = {json.dumps(items)};
    const STORAGE_KEY = {json.dumps(storage_key)};
    const DOWNLOAD_NAME = {json.dumps(download_name)};

    function loadResponses() {{
      try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }}
      catch (_) {{ return {{}}; }}
    }}

    function saveResponses(responses) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(responses));
      updateProgress();
    }}

    function setChoice(itemId, choice) {{
      const responses = loadResponses();
      responses[itemId] = choice;
      saveResponses(responses);
    }}

    function updateProgress() {{
      const responses = loadResponses();
      const done = ITEMS.filter(item => responses[item.item_id]).length;
      document.getElementById("progress").textContent = `${{done}} / ${{ITEMS.length}} annotated`;
    }}

    function render() {{
      const responses = loadResponses();
      const root = document.getElementById("items");
      root.innerHTML = "";
      ITEMS.forEach((item, idx) => {{
        const div = document.createElement("section");
        div.className = "item";
        div.innerHTML = `
          <div class="meta">${{item.display_index}}. ${{item.item_id}} · ${{item.pair_key}} · selector=${{item.selector}}</div>
          <img src="${{item.panel}}" alt="${{item.item_id}}">
          <div class="choices">
            ${{["A", "B", "Tie"].map(choice => `
              <label class="choice">
                <input type="radio" name="${{item.item_id}}" value="${{choice}}" ${{responses[item.item_id] === choice ? "checked" : ""}}>
                ${{choice}}
              </label>
            `).join("")}}
          </div>
        `;
        root.appendChild(div);
        div.querySelectorAll("input").forEach(input => {{
          input.addEventListener("change", () => setChoice(item.item_id, input.value));
        }});
      }});
      updateProgress();
    }}

    function jsonlText() {{
      const responses = loadResponses();
      return ITEMS
        .filter(item => responses[item.item_id])
        .map(item => JSON.stringify({{ item_id: item.item_id, choice: responses[item.item_id] }}))
        .join("\\n") + "\\n";
    }}

    function exportJsonl() {{
      const text = jsonlText();
      document.getElementById("jsonl").value = text;
      const blob = new Blob([text], {{ type: "application/jsonl" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = DOWNLOAD_NAME;
      a.click();
      URL.revokeObjectURL(url);
    }}

    async function copyJsonl() {{
      const text = jsonlText();
      document.getElementById("jsonl").value = text;
      await navigator.clipboard.writeText(text);
    }}

    function clearResponses() {{
      if (!confirm("Clear all annotations stored in this browser?")) return;
      localStorage.removeItem(STORAGE_KEY);
      render();
      document.getElementById("jsonl").value = "";
    }}

    render();
  </script>
</body>
</html>
"""

    with open(args.out, "w") as f:
        f.write(html)
    print(
        f"Wrote {args.out} with {len(items)} items "
        f"(manifest indices {args.start_index}:{subset_end})"
    )


if __name__ == "__main__":
    main()
