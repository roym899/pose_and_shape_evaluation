"""This module defines ShAPO interface.

Method is described in ShAPO: Implicit Representations for Multi-Object Shape,
Appearance and Pose Optimization, Irshad, 2022.

Implementation based on
[https://github.com/zubair-irshad/shapo](https://github.com/zubair-irshad/shapo).
"""
import os

from . import _shapo as shapo

class ShAPO(CPASMethod):
    """Wrapper class for ShAPO."""

    class Config(TypedDict):
        """Configuration dictionary for ShAPO.

        Attributes:

        """

    default_config: Config = {
    }

    def __init__(self, config: Config, camera: camera_utils.Camera) -> None:
        """Initialize and load SGPA model.

        Args:
            config: SGPA configuration. See SGPA.Config for more information.
            camera: Camera used for the input image.
        """
        config = yoco.load_config(config, current_dict=SGPA.default_config)
        self._parse_config(config)
        self._camera = camera
