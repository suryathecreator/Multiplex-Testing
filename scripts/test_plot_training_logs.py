#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import plot_training_logs


class PlotTrainingLogsTests(unittest.TestCase):
    def test_parse_console_logs_and_write_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            discrete_log = tmp_path / "discrete.log"
            multiplex_log = tmp_path / "multiplex.log"
            out_dir = tmp_path / "plots"
            summary_path = out_dir / "summary.json"
            discrete_log.write_text(
                "\n".join(
                    [
                        "step:0 - critic/rewards/mean:0.1 - response_length/mean:100 - actor/pg_loss:1.5 - actor/grad_norm:0.7 - perf/time_per_step:12.0",
                        "step:1 - critic/rewards/mean:0.2 - response_length/mean:90 - actor/pg_loss:1.2 - actor/grad_norm:0.6 - perf/time_per_step:11.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            multiplex_log.write_text(
                "\n".join(
                    [
                        "step:0 - critic/rewards/mean:0.3 - response_length/mean:80 - actor/pg_loss:1.1 - actor/grad_norm:0.5 - perf/time_per_step:13.0",
                        "step:1 - critic/rewards/mean:0.4 - response_length/mean:70 - actor/pg_loss:0.9 - actor/grad_norm:0.4 - perf/time_per_step:12.5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = plot_training_logs.parse_log(discrete_log)
            self.assertEqual([row["step"] for row in rows], [0, 1])
            self.assertEqual(rows[1]["critic/rewards/mean"], 0.2)

            plot_training_logs.main_from_args(
                [
                    "--run",
                    f"discrete_rl={discrete_log}",
                    "--run",
                    f"multiplex_thinking={multiplex_log}",
                    "--output-dir",
                    str(out_dir),
                    "--summary-json",
                    str(summary_path),
                ]
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["num_rows"]["discrete_rl"], 2)
            self.assertTrue((out_dir / "training_metrics_discrete_rl.csv").exists())
            self.assertTrue((out_dir / "training_reward_by_step.png").exists())
            self.assertTrue((out_dir / "training_combined_comparison.png").exists())


if __name__ == "__main__":
    unittest.main()
