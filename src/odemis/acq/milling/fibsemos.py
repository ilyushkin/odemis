# -*- coding: utf-8 -*-
"""
Created on 3 April 2025

@author: Patrick Cleeve

Copyright © 2025 Patrick Cleeve, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License version 2 as published by the Free
Software Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.
"""

import logging
import math
import os
import threading
import time
from concurrent import futures
from concurrent.futures._base import CANCELLED, FINISHED, RUNNING, CancelledError
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Union

from odemis import model
from odemis.acq.milling.patterns import (
    AsymmetricTrenchPatternParameters,
    MicroexpansionPatternParameters,
    MillingPatternParameters,
    RectanglePatternParameters,
    TrenchPatternParameters,
)

from odemis.acq.milling.tasks import (
    MillingSettings,
    MillingTaskSettings,
)
from odemis.acq.feature import CryoFeature, REFERENCE_IMAGE_FILENAME
from odemis.util import executeAsyncTask

# Check if fibsemOS is available
try:
    from fibsem.microscopes.odemis_microscope import (
        OdemisThermoMicroscope,
        OdemisTescanMicroscope,
        from_odemis_image
    )
    from fibsem.milling import (
        FibsemMillingStage,
        MillingAlignment,
        estimate_total_milling_time,
        mill_stages,
    )
    from fibsem.milling.patterning.patterns2 import (
        BasePattern,
        MicroExpansionPattern,
        RectanglePattern,
        TrenchPattern,
    )
    from fibsem.structures import (
        FibsemMillingSettings,
        FibsemRectangleSettings,
        Point,
        FibsemImage,
        FibsemImageMetadata,
        FibsemRectangle,
        BeamType,
        ImageSettings,
        MicroscopeState,
    )
    from fibsem.utils import load_microscope_configuration

    @dataclass
    class _AsymmetricTrenchPattern(BasePattern[FibsemRectangleSettings]):
        """fibsemOS pattern for an asymmetric trench with independent top and bottom rectangles.

        Both sub-rectangles are placed in a single DrawBeam layer so that
        DrawBeam.Start() is called only once, avoiding the Visibility (-4) error
        that occurs when two separate milling stages are used.

        The coordinate conventions match AsymmetricTrenchPatternParameters.generate():
        the top rectangle is centred above the gap midpoint, the bottom below it.
        """

        name: ClassVar[str] = "AsymmetricTrench"
        width_top: float = 10.0e-6
        height_top: float = 5.0e-6
        width_bottom: float = 10.0e-6
        height_bottom: float = 5.0e-6
        depth: float = 1.0e-6
        spacing: float = 5.0e-6

        def define(self) -> List[FibsemRectangleSettings]:
            """Return the two FibsemRectangleSettings that make up the asymmetric trench."""
            cx = self.point.x
            cy = self.point.y
            upper_cy = cy + self.spacing / 2 + self.height_top / 2
            lower_cy = cy - self.spacing / 2 - self.height_bottom / 2
            upper = FibsemRectangleSettings(
                width=self.width_top,
                height=self.height_top,
                depth=self.depth,
                centre_x=cx,
                centre_y=upper_cy,
                scan_direction="TopToBottom",
            )
            lower = FibsemRectangleSettings(
                width=self.width_bottom,
                height=self.height_bottom,
                depth=self.depth,
                centre_x=cx,
                centre_y=lower_cy,
                scan_direction="BottomToTop",
            )
            self.shapes = [upper, lower]
            return self.shapes

    FIBSEMOS_INSTALLED = True
except ImportError as e:
    logging.warning(f"fibsemOS is not installed or not available: {e}")
    FIBSEMOS_INSTALLED = False

_persistent_millmng: Optional["FibsemOSMillingTaskManager"] = None


def _get_reference_image(feature: CryoFeature) -> model.DataArray:
    """Get the in-memory reference image for a feature or raise."""

    if feature.reference_image is None:
        logging.error(
            "Missing reference image for feature '%s' (path=%s). "
            "This feature was likely loaded from disk without hydrating reference_image.",
            feature.name.value,
            getattr(feature, "path", None),
        )
        raise ValueError("Missing feature.reference_image.")
    return feature.reference_image


def _crop_to_reduced_area(ref_img: 'FibsemImage', rect: 'FibsemRectangle') -> 'FibsemImage':
    """Crop a fibsemOS image to the provided reduced-area rectangle.

    :param ref_img: The image to crop.
    :param rect: Rectangle with fractional coordinates (left, top, width, height).
    :return: The same image instance with cropped data.
    """

    h, w = ref_img.data.shape[-2], ref_img.data.shape[-1]

    # fractional to pixel indices
    x0 = int(rect.left * w)
    y0 = int(rect.top * h)
    x1 = int((rect.left + rect.width) * w)
    y1 = int((rect.top + rect.height) * h)

    # clamp to valid range just in case of rounding
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))

    # crop along the last two axes, DataArray slicing behaves like numpy
    ref_img.data = ref_img.data[..., y0:y1, x0:x1]
    return ref_img


def create_fibsemos_tfs_microscope() -> 'OdemisThermoMicroscope':
    """Create and return a fibsemOS Thermo microscope instance."""

    # TODO: extract the rest of the required metadata

    # stage metadata
    stage_bare = model.getComponent(role="stage-bare")
    stage_md = stage_bare.getMetadata()
    pre_tilt = stage_md[model.MD_CALIB].get(model.MD_SAMPLE_PRE_TILT, math.radians(35))
    rotation_reference = stage_md[model.MD_FAV_SEM_POS_ACTIVE]["rz"]

    # loads the default config
    config = load_microscope_configuration()
    config.system.stage.shuttle_pre_tilt = math.degrees(pre_tilt)
    # Used by fibsemOS for moving the stage flat to the electron beam
    config.system.stage.rotation_reference = math.degrees(rotation_reference)
    # Used by fibsemOS for moving the stage flat to the ion beam
    config.system.stage.rotation_180 = math.degrees(rotation_reference + math.pi)
    microscope = OdemisThermoMicroscope(config.system)

    return microscope

def create_fibsemos_tescan_microscope() -> 'OdemisTescanMicroscope':
    """Create, connect, and return a fibsemOS Tescan microscope instance."""

    # TODO: Extract the rest of the required metadata

    # stage metadata
    stage_bare = model.getComponent(role="stage-bare")
    stage_md = stage_bare.getMetadata()
    pre_tilt = stage_md[model.MD_CALIB].get(model.MD_SAMPLE_PRE_TILT, math.radians(35))
    rotation_reference = stage_md[model.MD_FAV_SEM_POS_ACTIVE]["rz"]

    # loads the default config
    config = load_microscope_configuration()
    config.system.stage.shuttle_pre_tilt = math.degrees(pre_tilt)
    # Used by fibsemOS for moving the stage flat to the electron beam
    config.system.stage.rotation_reference = math.degrees(rotation_reference)
    # Used by fibsemOS for moving the stage flat to the ion beam
    config.system.stage.rotation_180 = math.degrees(rotation_reference + math.pi)

    # Get the Tescan SEM component to extract host and port info
    fibsem = model.getComponent(role="fibsem")
    ip_address: str = fibsem.host
    port: int = fibsem.port
    # Pass the IP address to the fibsemOS config as well
    config.system.info.ip_address = ip_address
    microscope = OdemisTescanMicroscope(config.system)

    microscope.connect_to_microscope(ip_address, port)

    return microscope

def create_fibsemos_microscope() -> Union['OdemisThermoMicroscope', 'OdemisTescanMicroscope']:
    """Create and return a fibsemOS microscope instance matching the detected stage version."""
    stage_bare = model.getComponent(role="stage-bare")
    stage_md = stage_bare.getMetadata()
    md_calib = stage_md.get(model.MD_CALIB, {})
    stage_version = md_calib.get("version", None)

    if stage_version == "tfs_3":
        return create_fibsemos_tfs_microscope()
    elif stage_version == "tescan_1":
        return create_fibsemos_tescan_microscope()
    else:
        raise ValueError(f"Stage version {stage_version} is not supported")


def convert_pattern_to_fibsemos(p: MillingPatternParameters) -> 'BasePattern':
    """Convert from an Odemis pattern to a fibsemOS pattern"""
    if isinstance(p, RectanglePatternParameters):
        return _convert_rectangle_pattern(p)

    elif isinstance(p, TrenchPatternParameters):
        return _convert_trench_pattern(p)

    elif isinstance(p, AsymmetricTrenchPatternParameters):
        return _convert_asymmetric_trench_pattern(p)

    elif isinstance(p, MicroexpansionPatternParameters):
        return _convert_microexpansion_pattern(p)
    else:
        raise NotImplementedError(f"Conversion not implemented for pattern type: {type(p)}")

def _convert_rectangle_pattern(p: RectanglePatternParameters) -> 'RectanglePattern':
    """Convert an Odemis rectangle pattern to a fibsemOS RectanglePattern."""
    return RectanglePattern(
        width=p.width.value,
        height=p.height.value,
        depth=p.depth.value,
        rotation=p.rotation.value,
        scan_direction=p.scan_direction.value,
        point=Point(x=p.center.value[0], y=p.center.value[1])
    )

def _convert_trench_pattern(p: TrenchPatternParameters) -> 'TrenchPattern':
    """Convert an Odemis trench pattern to a fibsemOS TrenchPattern."""
    return TrenchPattern(
        width=p.width.value,
        upper_trench_height=p.height.value,
        lower_trench_height=p.height.value,
        spacing=p.spacing.value,
        depth=p.depth.value,
        point=Point(x=p.center.value[0], y=p.center.value[1])
    )

def _convert_asymmetric_trench_pattern(p: AsymmetricTrenchPatternParameters) -> '_AsymmetricTrenchPattern':
    """Convert an Odemis asymmetric trench pattern to a fibsemOS pattern.

    Returns a single _AsymmetricTrenchPattern whose define() method yields two
    FibsemRectangleSettings in one DrawBeam layer, so DrawBeam.Start() is
    called only once and the Visibility (-4) error is avoided.
    """
    return _AsymmetricTrenchPattern(
        width_top=p.width_top.value,
        height_top=p.height_top.value,
        width_bottom=p.width_bottom.value,
        height_bottom=p.height_bottom.value,
        depth=p.depth.value,
        spacing=p.spacing.value,
        point=Point(x=p.center.value[0], y=p.center.value[1]),
    )

def _convert_microexpansion_pattern(p: MicroexpansionPatternParameters) -> 'MicroExpansionPattern':
    """Convert an Odemis microexpansion pattern to a fibsemOS MicroExpansionPattern."""
    return MicroExpansionPattern(
        width=p.width.value,
        height=p.height.value,
        depth=p.depth.value,
        distance=p.spacing.value,
        point=Point(x=p.center.value[0], y=p.center.value[1])
    )

def _format_preset(voltage: float, current: float) -> str:
    """
    Format voltage (V) and current (A) into a preset name string like '30 keV; 150 pA'.
    Voltage is shown in eV below 1 keV, keV otherwise. Current is shown in pA, nA, or uA.

    This format is a convention for Tescan preset names. Presets in the Tescan
    software must be named following this convention so that fibsemOS can match
    them correctly.

    :param voltage: beam voltage in volts (must be a positive finite number).
    :param current: beam current in amperes (must be a positive finite number).
    :raises ValueError: if voltage or current are not positive finite numbers.
    """
    if not math.isfinite(voltage):
        raise ValueError(f"Voltage must be a finite number, got {voltage!r}")
    if voltage <= 0:
        raise ValueError(f"Voltage must be positive, got {voltage!r}")
    if not math.isfinite(current):
        raise ValueError(f"Current must be a finite number, got {current!r}")
    if current <= 0:
        raise ValueError(f"Current must be positive, got {current!r}")

    # Voltage: choose eV or keV based on order of magnitude
    if voltage < 1000:
        # eV range: [0, 1 keV)
        voltage_str = f"{voltage:g} eV"
    else:
        # keV range: [1 keV, ...)
        voltage_str = f"{voltage / 1000:g} keV"

    # Current: pA, nA, or uA based on order of magnitude
    if current < 1e-9:
        # pA range: [0, 1 nA)
        current_val = current * 1e12
        unit = "pA"
    elif current < 1e-6:
        # nA range: [1 nA, 1 uA)
        current_val = current * 1e9
        unit = "nA"
    else:
        # uA range: [1 uA, ...)
        current_val = current * 1e6
        unit = "uA"
    current_str = f"{current_val:g} {unit}"
    return f"{voltage_str}; {current_str}"

def convert_milling_settings(s: MillingSettings) -> 'FibsemMillingSettings':
    """Convert Odemis milling settings to fibsemOS milling settings.

    Both milling_current/milling_voltage and preset are populated because
    fibsemOS uses them selectively depending on the microscope backend:
    milling_current and milling_voltage are used by the TFS backend, while
    preset (a human-readable string such as "30 keV; 150 pA") is used by the
    Tescan backend.
    """
    return FibsemMillingSettings(
        milling_current=s.current.value,
        milling_voltage=s.voltage.value,
        patterning_mode=s.mode.value,
        hfw=s.field_of_view.value,
        preset=_format_preset(s.voltage.value, s.current.value)
    )

# task converter
def convert_task_to_milling_stage(task: MillingTaskSettings) -> List['FibsemMillingStage']:
    """Convert a single Odemis milling task to one or more fibsemOS milling stages.

    Patterns that have a native fibsemOS equivalent (Rectangle, Trench,
    MicroExpansion) are forwarded directly.  Any pattern whose type is not
    natively supported raises NotImplementedError in convert_pattern_to_fibsemos.
    In that case the pattern is expanded via generate() and each sub-shape is
    converted individually through the same dispatch.  Sub-shapes may be any
    natively supported primitive type (rectangle, circle, ...).

    This means that future compound patterns require only a correct generate()
    implementation and, if a new primitive type is introduced, a corresponding
    entry in convert_pattern_to_fibsemos no further changes here.

    :param task: the milling task to convert.
    :return: list of fibsemOS milling stages (one or more).
    """
    s = convert_milling_settings(task.milling)
    a = MillingAlignment(enabled=task.milling.align.value)
    stages = []

    for pattern in task.patterns:
        try:
            # Use native fibsemOS pattern type where available.
            # This keeps compound patterns (e.g. TrenchPattern) in a single
            # milling stage / DrawBeam layer so DrawBeam.Start() is called
            # only once for the whole pattern, avoiding visibility errors that
            # arise when sub-shapes are milled in separate layers.
            stages.append(FibsemMillingStage(
                name=task.name,
                milling=s,
                pattern=convert_pattern_to_fibsemos(pattern),
                alignment=a,
            ))
        except NotImplementedError:
            # No native fibsemOS equivalent — expand into rectangle sub-shapes
            # using Odemis coordinates and convert each one individually.
            # Sub-shapes from generate() are always RectanglePatternParameters.
            for sub_shape in pattern.generate():
                stages.append(FibsemMillingStage(
                    name=f"{task.name} - {sub_shape.name.value}",
                    milling=s,
                    pattern=convert_pattern_to_fibsemos(sub_shape),
                    alignment=a,
                ))

    return stages

def convert_milling_tasks_to_milling_stages(milling_tasks: List[MillingTaskSettings]) -> List['FibsemMillingStage']:
    """Convert a list of Odemis milling tasks to fibsemOS milling stages."""
    milling_stages = []
    for task in milling_tasks:
        milling_stages.extend(convert_task_to_milling_stage(task))
    return milling_stages

class FibsemOSMillingTaskManager:
    """Manage running milling tasks via fibsemOS using a persistent microscope connection."""

    def __init__(self):
        """Initialize the manager and establish the fibsemOS microscope connection."""
        # create microscope connection
        self.microscope = create_fibsemos_microscope()
        self._lock = threading.Lock()
        self._active = False
        self._cancel_requested = False

        # per-run state (set in async_run)
        self.milling_stages: List["FibsemMillingStage"] = []
        self._future: Optional[futures.Future] = None

    def cancel(self, future: futures.Future) -> bool:
        """Request cancellation of the current milling run."""
        logging.debug("Canceling milling procedure...")
        with self._lock:
            if not self._active:
                return False
            if self._cancel_requested:
                return True
            self._cancel_requested = True
        # Do not hold the lock during potentially blocking calls
        subf = getattr(future, "running_subf", None)
        if subf is not None:
            subf.cancel()
        try:
            self.microscope.stop_milling()
        finally:
            logging.debug("Milling procedure cancelled.")
        return True

    def estimate_milling_time(self) -> float:
        """Estimate the total milling time for the currently configured stages (seconds)."""
        return estimate_total_milling_time(self.milling_stages)

    def _run(self):
        """Internal worker that performs the milling stages sequentially."""
        future = self._future
        if future is None:
            # Should never happen if async_run configured correctly, but don't use assert.
            with self._lock:
                self._active = False
            raise RuntimeError("Internal error: milling run started without an associated future.")

        try:
            for stage in self.milling_stages:
                with self._lock:
                    if self._cancel_requested:
                        raise CancelledError()

                logging.info(f"Running milling stage: {stage.name}")
                ref_img = from_odemis_image(_get_reference_image(self.feature))
                ref_img.metadata.image_settings.path = self.path
                ref_img.metadata.image_settings.reduced_area = stage.alignment.rect

                ref_img = _crop_to_reduced_area(ref_img, stage.alignment.rect)

                mill_stages(self.microscope, [stage], ref_img)

        finally:
            with self._lock:
                self._active = False
                self._cancel_requested = False

    def async_run(self,
                  *,
                  future: futures.Future,
                  tasks: List[MillingTaskSettings],
                  feature: CryoFeature,
                  path: Optional[str] = None) -> futures.Future:
        """Prepare and start a milling run asynchronously (one run at a time)."""
        if path is None:
            path = os.getcwd()

        milling_stages = convert_milling_tasks_to_milling_stages(tasks)
        end_time = time.time() + estimate_total_milling_time(milling_stages) + 30

        with self._lock:
            if self._active:
                raise RuntimeError("A fibsemOS milling session is already running.")
            self._active = True
            self._cancel_requested = False
            self.microscope._last_imaging_settings.path = path
            self.milling_stages = milling_stages
            self.path = path
            self.feature = feature
            self._future = future
            self._future.running_subf = model.InstantaneousFuture()
            self._future.task_canceller = self.cancel
            # +30 s as estimate time only includes milling time, not current switching time, etc
            self._future.set_end_time(end_time)

            try:
                executeAsyncTask(self._future, self._run)
            except Exception:
                self._active = False
                raise
        return self._future


def run_milling_tasks_fibsemos(tasks: List[MillingTaskSettings], feature: CryoFeature, path: Optional[str] = None) -> futures.Future:
    """Run milling tasks asynchronously via fibsemOS.

    :param tasks: Milling tasks to execute (converted to fibsemOS stages and run in order).
    :param feature: Feature providing the in-memory reference image used for alignment.
        Must have ``feature.reference_image`` set.
    :param path: Directory where acquired reference images will be stored.
        Defaults to the current working directory.
    :return: A progressive future for progress reporting and cancellation.
    """
    global _persistent_millmng

    if _persistent_millmng is None:
        _persistent_millmng = FibsemOSMillingTaskManager()

    future = model.ProgressiveFuture()
    return _persistent_millmng.async_run(future=future, tasks=tasks, feature=feature, path=path)
