import sys
import unittest
from pathlib import Path

from scripts.run_osworld_verified import build_osworld_command, build_osworld_config


class RunOSWorldVerifiedTests(unittest.TestCase):
    def test_builds_single_env_osworld_command(self):
        env = {
            "AGENT_MODEL": "openai/gpt-5.4",
            "OSWORLD_REPO_DIR": ".external/OSWorld",
            "OSWORLD_PROVIDER_NAME": "docker",
            "OSWORLD_OBSERVATION_TYPE": "screenshot",
            "OSWORLD_CLIENT_PASSWORD": "password",
            "MAX_STEPS": "3",
            "OUTPUT_DIR": "outputs/runs",
        }

        config = build_osworld_config(env=env, now="2026-06-25_131600")
        command = build_osworld_command(config, help_text="--provider_name\n--observation_type\n--model\n--max_steps\n--result_dir\n--headless")

        self.assertEqual(config.benchmark, "osworld-verified")
        self.assertEqual(config.run_id, "2026-06-25_131600_openai_gpt-5.4_osworld-verified")
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(Path(command[1]).name, "run.py")
        self.assertIn("--provider_name", command)
        self.assertIn("docker", command)
        self.assertIn("--observation_type", command)
        self.assertIn("screenshot", command)
        self.assertIn("--model", command)
        self.assertIn("openai/gpt-5.4", command)
        self.assertIn("--max_steps", command)
        self.assertIn("3", command)
        self.assertIn("--result_dir", command)
        self.assertNotIn("--client_password", command)

    def test_can_target_one_osworld_domain_and_example(self):
        env = {
            "AGENT_MODEL": "openai/gpt-5.4-mini",
            "OSWORLD_REPO_DIR": ".external/OSWorld",
            "OSWORLD_DOMAIN": "libreoffice_impress",
            "OSWORLD_EXAMPLE_ID": "a669ef01-ded5-4099-9ea9-25e99b569840",
            "OUTPUT_DIR": "outputs/runs",
        }

        config = build_osworld_config(env=env, now="2026-06-25_131600")
        command = build_osworld_command(config, help_text="--provider_name\n--observation_type\n--model\n--max_steps\n--result_dir\n--domain")

        self.assertIn("--domain", command)
        self.assertIn("libreoffice_impress", command)
        self.assertNotIn("--example_id", command)


if __name__ == "__main__":
    unittest.main()
