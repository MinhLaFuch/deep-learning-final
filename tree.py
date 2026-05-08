import os
import sys

def print_tree(root, prefix="", max_files=5, lines=None):
    entries = sorted(os.scandir(root), key=lambda e: (e.is_file(), e.name))
    dirs    = [e for e in entries if e.is_dir()]
    files   = [e for e in entries if e.is_file()]

    all_entries = dirs + files[:max_files]
    hidden      = len(files) - max_files if len(files) > max_files else 0

    for i, entry in enumerate(all_entries):
        is_last   = (i == len(all_entries) - 1) and hidden == 0
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + entry.name)
        if entry.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(entry.path, prefix + extension, max_files, lines)

    if hidden > 0:
        lines.append(prefix + f"└── ... and {hidden} more files")


root    = sys.argv[1] if len(sys.argv) > 1 else "."
output  = sys.argv[2] if len(sys.argv) > 2 else "tree.txt"

lines = [root]
print_tree(root, lines=lines)

text = "\n".join(lines)
print(text)

with open(output, "w", encoding="utf-8") as f:
    f.write(text)

print(f"\nSaved to {output}")