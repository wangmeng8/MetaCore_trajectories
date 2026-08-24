import json
import tempfile
import unittest
from pathlib import Path

from scripts.runner_common import write_json


class RunnerCommonTests(unittest.TestCase):
    def test_write_json_serializes_bytes_as_base64_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"

            write_json(path, {"raw": {"image_preview": {"bytes": b"abc", "path": None}}})

            payload = json.loads(path.read_text(encoding="utf-8"))
            stored = payload["raw"]["image_preview"]["bytes"]
            self.assertEqual(stored["__type__"], "bytes")
            self.assertEqual(stored["encoding"], "base64")
            self.assertEqual(stored["length"], 3)
            self.assertEqual(stored["data"], "YWJj")


if __name__ == "__main__":
    unittest.main()
