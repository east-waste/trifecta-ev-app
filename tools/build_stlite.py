"""GitHub Pages 公開用の stlite (ブラウザ内 Streamlit) ページを生成する。

app.py と src/ 配下の Python ファイルを読み込み、docs/index.html に埋め込む。
生成後、GitHub Pages の公開元を「main ブランチ / docs フォルダ」にすると、
    https://<ユーザー名>.github.io/<リポジトリ名>/
で Streamlit アプリがブラウザだけで動く。

使い方:
    python3 tools/build_stlite.py
"""

from __future__ import annotations

import json
from pathlib import Path

# stlite (@stlite/browser) のバージョン。必要に応じて更新する。
STLITE_VERSION = "0.85.1"

# 仮想ファイルシステムに載せる Python ファイル（リポジトリルートからの相対パス）。
PYTHON_FILES = [
    "app.py",
    "src/__init__.py",
    "src/calculator.py",
    "src/optimizer.py",
    "src/estimator.py",
    "src/recommendation.py",
    "src/ocr.py",
    "src/validation.py",
    "src/sample_data.py",
]

# Pyodide 上で追加インストールが必要なパッケージ（streamlit は stlite に同梱）。
REQUIREMENTS = ["pandas", "numpy"]

ENTRYPOINT = "app.py"
PAGE_TITLE = "三連複 2倍回収・期待値最大化シミュレーター"


HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <title>{title}</title>
    <link
      rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/@stlite/browser@{version}/build/stlite.css"
    />
    <style>
      #loading {{
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        margin: 24px;
        color: #333;
      }}
    </style>
  </head>
  <body>
    <div id="root">
      <div id="loading">
        アプリを読み込み中です（初回はPython環境のダウンロードに数十秒かかることがあります）…
      </div>
    </div>
    <script id="app-files" type="application/json">{files_json}</script>
    <script type="module">
      import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{version}/build/stlite.js";
      const files = JSON.parse(document.getElementById("app-files").textContent);
      mount(
        {{
          requirements: {requirements_json},
          entrypoint: "{entrypoint}",
          files: files,
        }},
        document.getElementById("root"),
      );
    </script>
  </body>
</html>
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    files: dict[str, str] = {}
    for rel_path in PYTHON_FILES:
        path = root / rel_path
        files[rel_path] = path.read_text(encoding="utf-8")

    files_json = json.dumps(files, ensure_ascii=False)
    # <script> タグ内に安全に埋め込むため、終了タグ類をエスケープする。
    files_json = files_json.replace("</", "<\\/")

    html = HTML_TEMPLATE.format(
        title=PAGE_TITLE,
        version=STLITE_VERSION,
        files_json=files_json,
        requirements_json=json.dumps(REQUIREMENTS),
        entrypoint=ENTRYPOINT,
    )

    docs_dir = root / "docs"
    docs_dir.mkdir(exist_ok=True)
    # GitHub Pages が Jekyll 処理をしないようにする（_ で始まる名前等を素通しする）。
    (docs_dir / ".nojekyll").write_text("", encoding="utf-8")
    (docs_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"Generated: {docs_dir / 'index.html'}")
    print(f"Embedded files: {', '.join(files.keys())}")


if __name__ == "__main__":
    main()
