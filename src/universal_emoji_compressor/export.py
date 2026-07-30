"""将视觉回填后的 SQLite 重新导出为 JSONL。"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    connection = sqlite3.connect(args.output_root / "index.sqlite")
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT a.*,json_group_array(s.original_path) AS original_paths
        FROM assets a LEFT JOIN sources s ON s.asset_id=a.id
        WHERE a.status='ok' GROUP BY a.id ORDER BY a.id
        """
    )
    mapping = {
        "metrics_json": "metrics",
        "decision_json": "decision",
        "sampled_frames_json": "sampled_frames",
        "ocr_json": "ocr",
        "understanding_json": "understanding",
        "skipped_frame_indices_json": "skipped_frame_indices",
        "original_paths": "original_paths",
    }
    output = args.output_root / "index.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = dict(row)
            for old, new in mapping.items():
                record[new] = json.loads(record.pop(old) or "null")
            record["recovered"] = bool(record.get("recovered"))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    connection.close()
    print(output)


if __name__ == "__main__":
    main()
