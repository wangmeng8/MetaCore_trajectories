import unittest

from scripts.run_tau2_gpt55 import build_run_config


class RunTau2DataDirTests(unittest.TestCase):
    def test_uses_tau2_data_dir_for_simulation_scan_default(self):
        config = build_run_config(
            env={
                "OUTPUT_DIR": "outputs/runs",
                "TAU2_DATA_DIR": ".external/tau2-bench/data",
            },
            now="2026-06-24_120000",
        )

        self.assertEqual(str(config.simulations_dir), ".external\\tau2-bench\\data\\simulations")

    def test_uses_tau2_repo_dir_for_simulation_scan_default(self):
        config = build_run_config(
            env={
                "OUTPUT_DIR": "outputs/runs",
                "TAU2_REPO_DIR": ".external/tau2-bench",
            },
            now="2026-06-24_120000",
        )

        self.assertEqual(str(config.simulations_dir), ".external\\tau2-bench\\data\\simulations")


if __name__ == "__main__":
    unittest.main()
