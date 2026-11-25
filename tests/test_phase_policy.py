"""Checks for phase features and reproducible development demand."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from phase_policy import FEATURES, LEGACY_FEATURES, LearnedPhaseController
from train_phase_policy import FIT_SEEDS, REFINEMENT_SEEDS, ROUTES, VALIDATION_SEEDS, training_routes


class PhasePolicyTests(unittest.TestCase):
    def test_approaching_vehicle_can_change_the_preferred_phase(self):
        positions = {"north": 45, "east": 90}
        lanes = SimpleNamespace(
            getMaxSpeed=lambda lane: 10,
            getLastStepVehicleIDs=lambda lane: [lane],
            getLastStepHaltingNumber=lambda lane: 0,
            getWaitingTime=lambda lane: 0,
        )
        signal = SimpleNamespace(
            lanes=["north", "east"], lanes_length={"north": 100, "east": 100},
            id="signal", green_phase=0, delta_time=5,
            green_phases=[SimpleNamespace(state="Gr"), SimpleNamespace(state="rG")],
            sumo=SimpleNamespace(
                lane=lanes,
                trafficlight=SimpleNamespace(
                    getControlledLinks=lambda _: [[("north",)], [("east",)]],
                ),
                vehicle=SimpleNamespace(getLanePosition=positions.__getitem__),
            ),
        )
        weights = [float(name == "approaching") for name in FEATURES]
        policy = LearnedPhaseController(weights)
        self.assertEqual(policy.choose_phase(signal, can_switch=False), 0)
        self.assertEqual(policy.choose_phase(signal, can_switch=True), 1)
        positions["north"] = 95
        self.assertEqual(policy.choose_phase(signal, can_switch=True), 0)

    def test_loads_legacy_checkpoint_in_feature_order(self):
        weights = dict(zip(LEGACY_FEATURES, range(len(LEGACY_FEATURES))))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "phase.json"
            path.write_text(json.dumps(weights, sort_keys=True), encoding="utf-8")
            with patch("phase_policy.MODEL_PATH", path):
                policy = LearnedPhaseController()
        self.assertEqual(policy.weights.tolist(), list(weights.values()))

    def test_development_demand_is_reproducible_and_held_out(self):
        development = (*FIT_SEEDS, *VALIDATION_SEEDS, *REFINEMENT_SEEDS)
        self.assertEqual(len(set(development)), len(development))
        self.assertTrue(set(development).isdisjoint(range(1101, 1109)))
        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            routes = training_routes(Path(first), FIT_SEEDS[0], 250)
            repeat = training_routes(Path(second), FIT_SEEDS[0], 250)
            self.assertEqual(routes[:2], ROUTES)
            self.assertEqual([file.read_bytes() for file in routes[2:]],
                             [file.read_bytes() for file in repeat[2:]])
            for path in routes[2:]:
                root = ET.parse(path).getroot()
                self.assertEqual(len(root.findall("route")), 12)
                flows = root.findall("flow")
                self.assertEqual(len(flows), 24)
                self.assertTrue(all(flow.get("period").startswith("exp(") for flow in flows))
                self.assertEqual(max(int(flow.get("end")) for flow in flows), 250)
            with self.assertRaises(ValueError):
                training_routes(Path(first), 1101, 250)
