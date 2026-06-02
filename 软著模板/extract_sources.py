#!/usr/bin/env python3
"""
源代码提取脚本 — 软著申请用
输出: 源代码文档.txt
"""
import os
from pathlib import Path

PROJECT_ROOT = Path("/home/ysdhanji/ocrcplus")
OUTPUT_FILE = Path("/home/ysdhanji/ocrcplus/软著模板/源代码文档.txt")

INCLUDE_FILES = [
    "server.py", "start.sh", "stress_test.py",
    "static/index.html", "requirements.txt",
]

INCLUDE_DIRS = [
    "PaddleOCR/deploy/cpp_infer/src",
]

EXTRA_FILES = [
    "PaddleOCR/deploy/cpp_infer/cli.cc",
    "PaddleOCR/deploy/cpp_infer/CMakeLists.txt",
]

EXCLUDE = ["build", "build_trt", "third_party", ".git", "venv", "models",
           "paddle_inference_install_dir", "__pycache__", "backup", ".pyc", ".o", ".so"]


def should_exclude(path: str) -> bool:
    parts = path.replace(str(PROJECT_ROOT), "").split("/")
    for ex in EXCLUDE:
        if ex in parts: return True
    return False


def collect() -> list:
    files = []
    for f in INCLUDE_FILES:
        p = PROJECT_ROOT / f
        if p.exists() and not should_exclude(str(p)):
            files.append(p)
    for d in INCLUDE_DIRS + EXTRA_FILES:
        p = PROJECT_ROOT / d
        if p.is_file() and not should_exclude(str(p)):
            files.append(p)
        elif p.is_dir():
            for root, dirs, fnames in os.walk(p):
                dirs[:] = [x for x in dirs if x not in EXCLUDE]
                for fn in sorted(fnames):
                    fp = Path(root) / fn
                    if not should_exclude(str(fp)):
                        files.append(fp)
    return sorted(set(files))


def generate():
    files = collect()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for fp in files:
            rel = fp.relative_to(PROJECT_ROOT)
            out.write(f"\n// ===== {rel} =====\n\n")
            try:
                content = fp.read_text(encoding="utf-8")
                out.write(content)
            except Exception as e:
                out.write(f"// Error: {e}\n")
            out.write("\n")
    total = sum(1 for _ in open(OUTPUT_FILE, encoding="utf-8"))
    print(f"✅ 提取完成: {OUTPUT_FILE}")
    print(f"   文件数: {len(files)}, 总行数: {total}")


if __name__ == "__main__":
    generate()
