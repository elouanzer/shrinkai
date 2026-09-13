"""Generates the API reference pages and navigation.

Runs on every `mkdocs build`/`mkdocs serve` (wired in `mkdocs.yml` via the
`gen-files` plugin). Walks every `.py` file under `src/shrinkai/`, creates one
Markdown page per module under `docs/reference/`, and writes
`docs/reference/SUMMARY.md` so `mkdocs-literate-nav` can build a navigation
tree that mirrors the package structure — drilling down module by module, the
same way scikit-learn's or numpy's API reference does. Nothing needs to be
maintained by hand: new/renamed/removed modules just show up automatically on
the next build.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

repo_root = Path(__file__).parent.parent
package_dir = repo_root / "src" / "shrinkai"

for path in sorted(package_dir.rglob("*.py")):
    module_path = path.relative_to(package_dir).with_suffix("")
    doc_path = path.relative_to(package_dir).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    nav_parts = list(module_path.parts)
    identifier_parts = ["shrinkai", *module_path.parts]

    if nav_parts and nav_parts[-1] == "__init__":
        nav_parts = nav_parts[:-1]
        identifier_parts = identifier_parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif nav_parts and nav_parts[-1] == "__main__":
        continue

    if not nav_parts:
        # The top-level `shrinkai/__init__.py` itself: covered by docs/index.md.
        continue

    nav[nav_parts] = doc_path.as_posix()

    identifier = ".".join(identifier_parts)
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        print(f"::: {identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(repo_root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.write("* [Overview](index.md)\n")
    nav_file.writelines(nav.build_literate_nav())
