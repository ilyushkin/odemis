# -*- coding: utf-8 -*-
"""
Created on Oct 2021

@author: Alexéy Ilyushkin

Copyright © 2021-2026 Alexéy Ilyushkin, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License version 2 as published by the Free Software
Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.
"""

import logging
import os
import random
import unittest

import numpy

from odemis import model
from odemis.acq.feature import (
    CryoFeature,
    DEFAULT_MILLING_ALIGNMENT_AREA,
    FEATURE_READY_TO_MILL,
    MIN_MILLING_ALIGNMENT_AREA_PIXELS,
    MillingAlignmentAreaTooSmallError,
    REFERENCE_IMAGE_FILENAME,
    constrain_milling_alignment_area,
    feature_decoder,
    load_milling_tasks,
)
from odemis.acq.milling import DEFAULT_MILLING_TASKS_PATH
from odemis.acq.move import Posture

logging.getLogger().setLevel(logging.DEBUG)

# store the test-features as json for easier editting
TEST_FEATURES_PATH = os.path.join(os.path.dirname(__file__), "test-features.json")
with open(TEST_FEATURES_PATH, "r") as f:
    TEST_FEATURES_STR = f.read()

class TestFeatureEncoderDecoder(unittest.TestCase):
    """
    Test the json encoder and decoder of the CryoFeature class
    """
    path = ""

    def tearDown(self):
        if os.path.exists(self.path):
            filename = os.path.join(self.path, f"TestFeature-1-{REFERENCE_IMAGE_FILENAME}")
            if os.path.exists(filename):
                os.remove(filename)
            os.rmdir(self.path)

    def test_notch_feature_offset(self):
        """Keep the notch right of the centered polishing pattern."""
        tasks = load_milling_tasks(DEFAULT_MILLING_TASKS_PATH)
        feature = CryoFeature(
            name="TestFeature-1",
            stage_position={"x": 0, "y": 0},
            fm_focus_position={"z": 0},
            milling_tasks={
                "Notch": tasks["Notch"],
                "Polishing 02": tasks["Polishing 02"],
            },
        )
        feature_position = (12e-6, -8e-6)

        feature.set_milling_feature_offset(feature_position)

        notch = feature.milling_tasks["Notch"].patterns[0]
        polishing = feature.milling_tasks["Polishing 02"].patterns[0]
        for actual, expected in zip(notch.center.value, (15e-6, -8e-6)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(polishing.center.value, feature_position)
        notch.mirrored.value = True
        feature.set_milling_feature_offset(feature_position)
        for actual, expected in zip(notch.center.value, (15e-6, -8e-6)):
            self.assertAlmostEqual(actual, expected)

    def test_decoder_ignores_unsupported_milling_tasks(self):
        """Ignore saved tasks that contain no patterns supported by this version."""
        feature = feature_decoder({
            "name": "Future feature",
            "status": FEATURE_READY_TO_MILL,
            "stage_position": {"x": 0, "y": 0},
            "fm_focus_position": {"z": 0},
            "milling_tasks": {
                "Future task": {
                    "name": "Future task",
                    "milling": {
                        "current": 100e-9,
                        "voltage": 30e3,
                        "field_of_view": 400e-6,
                        "mode": "Serial",
                        "channel": "ion",
                    },
                    "patterns": [{"pattern": "future_pattern"}],
                },
            },
        })

        self.assertEqual(feature.milling_tasks, {})

    def test_feature_milling_tasks(self):
        feature = CryoFeature(
            name="TestFeature-1",
            stage_position={"x": 50e-6, "y": 25e-6, "z": 32e-3, "rx": 0.61, "rz": 0},
            fm_focus_position={"z": 1.69e-3}
        )
        stage_position = {"x": 25e-6, "y": 40e-6, "z": 32e-3, "rx": 0.31, "rz": 0}
        self.path = os.path.join(os.getcwd(), feature.name.value)
        reference_image = model.DataArray(numpy.zeros(shape=(1024, 1536)), metadata={})
        milling_tasks = load_milling_tasks(DEFAULT_MILLING_TASKS_PATH)
        milling_feature_offset = (12e-6, -8e-6)

        # randomly remove some milling tasks (to simulate user choice)
        task_name = random.choice(list(milling_tasks.keys()))
        del milling_tasks[task_name]

        # save milling task data
        feature.save_milling_task_data(
            stage_position=stage_position,
            path=self.path,
            reference_image=reference_image,
            milling_tasks=milling_tasks
        )
        feature.set_milling_feature_offset(milling_feature_offset)

        self.assertEqual(feature.path, self.path)
        self.assertEqual(feature.reference_image.shape, reference_image.shape)
        self.assertEqual(feature.get_posture_position(Posture.MILLING), stage_position)
        self.assertEqual(feature.milling_feature_offset.value, milling_feature_offset)
        for task in feature.milling_tasks.values():
            for pattern in task.patterns:
                self.assertEqual(
                    pattern.center.value,
                    pattern.get_center_at_feature(milling_feature_offset),
                )
        self.assertEqual(feature.status.value, FEATURE_READY_TO_MILL)
        self.assertEqual(set(feature.milling_tasks.keys()), set(milling_tasks.keys()))

        # assert directory and file is created
        self.assertTrue(os.path.exists(feature.path))

        filename = os.path.join(feature.path, f"{feature.name.value}-{REFERENCE_IMAGE_FILENAME}")
        self.assertTrue(os.path.exists(filename))

    def test_too_small_reference_image_is_not_saved(self):
        feature = CryoFeature(
            name="TestFeature-1",
            stage_position={"x": 0, "y": 0},
            fm_focus_position={"z": 0},
        )
        self.path = os.path.join(os.getcwd(), feature.name.value)
        reference_image = model.DataArray(numpy.zeros(shape=(200, 300)), metadata={})

        with self.assertRaises(MillingAlignmentAreaTooSmallError):
            feature.save_milling_task_data(
                stage_position={"x": 0, "y": 0},
                path=self.path,
                reference_image=reference_image,
            )

        self.assertFalse(os.path.exists(self.path))
        self.assertIsNone(feature.reference_image)


class TestMillingAlignmentArea(unittest.TestCase):
    def test_default_area(self):
        feature = CryoFeature("Feature-1", {"x": 0, "y": 0, "z": 0}, {"z": 0})

        self.assertEqual(feature.millingAlignmentArea.value, DEFAULT_MILLING_ALIGNMENT_AREA)

    def test_area_is_clamped_to_minimum_pixel_count_and_image_bounds(self):
        area = constrain_milling_alignment_area((0.9, 0.9, 0.1, 0.1), (1024, 2048))

        expected = (0.8125, 0.75, 0.1875, 0.25)
        for actual_value, expected_value in zip(area, expected):
            self.assertAlmostEqual(actual_value, expected_value)

    def test_square_reference_image_can_contain_minimum_area(self):
        area = constrain_milling_alignment_area(DEFAULT_MILLING_ALIGNMENT_AREA, (512, 512))

        pixel_width = area[2] * 512
        pixel_height = area[3] * 512
        self.assertAlmostEqual(pixel_width * pixel_height, MIN_MILLING_ALIGNMENT_AREA_PIXELS)
        self.assertAlmostEqual(pixel_width, pixel_height)

    def test_minimum_area_accepts_both_extreme_aspect_ratios(self):
        for area in ((0.0, 0.0, 0.75, 0.5), (0.0, 0.0, 0.5, 0.75)):
            with self.subTest(area=area):
                self.assertEqual(constrain_milling_alignment_area(area, (512, 512)), area)

    def test_area_constrains_narrow_shape_even_with_enough_pixels(self):
        cases = (
            ((0.0, 0.0, 0.75, 0.125), (0.0, 0.0, 0.75, 0.5)),
            ((0.0, 0.0, 0.125, 0.75), (0.0, 0.0, 0.5, 0.75)),
        )
        for area, expected in cases:
            with self.subTest(area=area):
                self.assertEqual(constrain_milling_alignment_area(area, (1024, 1024)), expected)

    def test_reference_image_must_contain_minimum_area(self):
        with self.assertRaises(ValueError):
            constrain_milling_alignment_area((0.1, 0.2, 0.25, 0.25), (200, 300))

    def test_feature_accepts_free_aspect_ratio(self):
        feature = CryoFeature("Feature-1", {"x": 0, "y": 0, "z": 0}, {"z": 0})

        feature.millingAlignmentArea.value = (0.1, 0.1, 0.3, 0.2)

        self.assertEqual(feature.millingAlignmentArea.value, (0.1, 0.1, 0.3, 0.2))


if __name__ == "__main__":
    unittest.main()
