"""Integration checks against SUMO's trip log and traffic-light API."""

import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET


@unittest.skipUnless(os.environ.get("SUMO_HOME"), "requires a local SUMO installation")
class EnvironmentTests(unittest.TestCase):
    def test_native_program_and_arrivals_match_trip_log(self):
        from unified_mdp_env import UnifiedMDPIntersectionEnv

        begin = 25000
        with tempfile.TemporaryDirectory(prefix="sumo trip log ") as directory:
            trips = Path(directory) / "arrivals.xml"
            program = Path(__file__).resolve().parents[1] / "actuated.add.xml"
            env = UnifiedMDPIntersectionEnv(
                fixed_ts=True, program_id="actuated", begin_time=begin, num_seconds=100,
                additional_sumo_cmd=["-a", program, "--tripinfo-output", trips],
            )
            try:
                # The evaluation starts a fresh episode on each reset.
                env.reset(seed=7)
                _, initial = env.reset(seed=7)
                self.assertEqual(initial["step"], begin)
                counts = []
                phases = set()
                while True:
                    _, _, _, done, info = env.step(None)
                    tls = env.env.sumo.trafficlight
                    self.assertEqual(tls.getProgram(env.ts_id), "actuated")
                    phases.add(tls.getPhase(env.ts_id))
                    counts.append(info["step_throughput"])
                    if done:
                        break
            finally:
                env.close()

            self.assertEqual(info["step"], begin + 100)
            self.assertGreater(len(phases), 1)
            expected = [0] * len(counts)
            for trip in ET.parse(trips).getroot().findall("tripinfo"):
                index = int((float(trip.attrib["arrival"]) - begin) // 5)
                expected[index] += 1
            self.assertGreater(sum(expected), 0)
            self.assertEqual(counts, expected)

    def test_mask_matches_the_native_yellow_and_minimum_green_guard(self):
        from unified_mdp_env import UnifiedMDPIntersectionEnv

        env = UnifiedMDPIntersectionEnv(num_seconds=20)
        try:
            observation, _ = env.reset(seed=7)
            self.assertEqual(observation.shape, (25,))
            self.assertEqual(env.action_space.n, 4)
            self.assertEqual(env.get_action_mask().tolist(), [True, False, False, False])
            env.step(1)
            self.assertEqual(env.get_action_mask().tolist(), [True, False, False, False])
            observation, *_ = env.step(1)
            self.assertTrue(env.get_action_mask().all())
            self.assertEqual(observation[16:20].tolist(), [1, 0, 0, 0])
            observation, *_ = env.step(1)
            self.assertEqual(observation[16:20].tolist(), [0, 1, 0, 0])
        finally:
            env.close()
