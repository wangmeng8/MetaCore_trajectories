import io
import unittest

from scripts.runner_common import ProgressPrinter, format_progress_bar, format_progress_line


class RunnerCommonProgressTests(unittest.TestCase):
    def test_formats_progress_bar_with_total(self):
        self.assertEqual(format_progress_bar(3, 10, width=10), "[###-------] 3/10 30.0%")

    def test_formats_progress_line_with_extra_context(self):
        line = format_progress_line(
            "HLE",
            completed=2,
            total=5,
            elapsed_seconds=65,
            extra="workers=4",
        )

        self.assertEqual(line, "HLE progress [########------------] 2/5 40.0% | elapsed 01:05 | workers=4")

    def test_progress_printer_throttles_and_allows_forced_final_line(self):
        stream = io.StringIO()
        now = [100.0]

        printer = ProgressPrinter(
            label="Tau2",
            total=None,
            interval_seconds=10,
            stream=stream,
            clock=lambda: now[0],
        )

        printer.update(0, extra="raw changed=0")
        printer.update(1, extra="raw changed=1")
        now[0] = 111.0
        printer.update(2, extra="raw changed=2")
        printer.update(3, extra="done", force=True)

        lines = stream.getvalue().strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "Tau2 progress 0 completed | elapsed 00:00 | raw changed=0")
        self.assertEqual(lines[1], "Tau2 progress 2 completed | elapsed 00:11 | raw changed=2")
        self.assertEqual(lines[2], "Tau2 progress 3 completed | elapsed 00:11 | done")


if __name__ == "__main__":
    unittest.main()
