"""Check the benchmark's horizon accounting against standalone SUMO statistics."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET


@unittest.skipUnless(os.environ.get("SUMO_HOME"), "requires a local SUMO installation")
class TripMetricsTests(unittest.TestCase):
    def test_completed_running_pending_and_future_trips(self):
        from sumolib import checkBinary
        from run_ablation_and_comparison import DEFAULT_ROUTE, evaluate_episode

        with tempfile.TemporaryDirectory(prefix="trip metrics ") as directory:
            folder = Path(directory)
            route = folder / "platoon.rou.xml"
            routes = ET.Element("routes")
            ET.SubElement(routes, "route", id="r", edges="n_t t_s")
            # One completed trip, then a platoon too large to insert before the cutoff.
            for index, depart in enumerate([0] + [78] * 20 + [100]):
                ET.SubElement(routes, "vehicle", id=str(index), route="r", depart=str(depart))
            ET.ElementTree(routes).write(route)
            result = evaluate_episode("fixed", None, 17, 85, 0, route)

            stats_file, trips_file = folder / "stats.xml", folder / "trips.xml"
            subprocess.run([
                checkBinary("sumo"), "-n", str(DEFAULT_ROUTE.with_name("single-intersection.net.xml")),
                "-r", str(route), "--seed", "17", "--end", "85", "--no-step-log",
                "--statistic-output", str(stats_file), "--tripinfo-output", str(trips_file),
                "--tripinfo-output.write-undeparted", "--tripinfo-output.write-unfinished",
            ], check=True, capture_output=True)
            stats = ET.parse(stats_file)
            vehicles = stats.find("vehicles").attrib
            trips = stats.find("vehicleTripStatistics").attrib
            self.assertEqual(result["scheduled_veh"], 21)
            self.assertEqual(result["arrivals_veh"], 1)
            self.assertGreater(int(vehicles["running"]), 0)
            self.assertGreater(int(vehicles["waiting"]), 0)
            self.assertEqual(result["pending_insert_veh"], int(vehicles["waiting"]))
            delay = float(trips["totalDepartDelay"])
            self.assertAlmostEqual(result["demand_elapsed_veh_s"], float(trips["totalTravelTime"]) + delay)
            self.assertAlmostEqual(result["demand_mean_depart_delay_s"], delay / 21)
            # SUMO rounds its aggregate mean timeLoss to two decimal places.
            expected_loss = float(trips["timeLoss"]) * int(trips["count"])
            self.assertAlmostEqual(result["demand_mean_delay_s"], (delay + expected_loss) / 21, delta=0.01)
            first = ET.parse(trips_file).find("tripinfo[@id='0']")
            self.assertGreater(float(first.attrib["arrival"]), 0)
            self.assertAlmostEqual(result["arrived_mean_time_loss_s"], float(first.attrib["timeLoss"]))
