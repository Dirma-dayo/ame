#!/usr/bin/env python3
"""
Scans assets/ame_sprite for animation folders where sorted filenames don't
match the numeric frame index embedded in the name - almost always caused by
leftover " #12345" export duplicates sitting next to the real numbered frame.

Usage:
    python3 tools/audit_frames.py [assets/ame_sprite]
"""
import re
import sys
from pathlib import Path

FRAME_RE = re.compile(r"_(\d+)(?:\s+#\d+)?\.png$")

def audit_folder(folder: Path):
    files = sorted(p.name for p in folder.glob("*.png"))
    if not files:
        return

    indices = []
    suspects = []
    for name in files:
        m = FRAME_RE.search(name)
        if not m:
            suspects.append((name, "doesn't match expected frame pattern"))
            continue
        idx = int(m.group(1))
        if " #" in name:
            suspects.append((name, f"looks like a duplicate/orphan export of frame {idx}"))
        indices.append(idx)

    seen = set()
    dupes = {i for i in indices if i in seen or seen.add(i)}
    if dupes:
        for name in files:
            m = FRAME_RE.search(name)
            if m and int(m.group(1)) in dupes and (name, "") not in [(n, "") for n, _ in suspects]:
                pass  # already caught above via " #" check in most cases

    if suspects:
        print(f"\n{folder}")
        for name, reason in suspects:
            print(f"  ! {name}  <- {reason}")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "assets/ame_sprite")
    if not root.exists():
        print(f"Can't find {root} (run this from your project root, or pass a path)")
        sys.exit(1)

    # Any directory containing at least one .png is treated as a leaf folder.
    leaf_dirs = sorted({p.parent for p in root.rglob("*.png")})
    for folder in leaf_dirs:
        audit_folder(folder)


if __name__ == "__main__":
    main()
