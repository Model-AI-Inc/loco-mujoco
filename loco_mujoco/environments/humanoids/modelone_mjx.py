import mujoco
from mujoco import MjSpec

from .modelone import ModelOne # Assuming modelone.py is in the same directory


class MjxModelOne(ModelOne):

    mjx_enabled = True

    def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=2, ls_iterations=4, disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]
        super().__init__(timestep=timestep, n_substeps=n_substeps, model_option_conf=model_option_conf, **kwargs)

    def _modify_spec_for_mjx(self, spec: MjSpec):
        foot_geoms = ["right_foot_1_col","left_foot_1_col", "right_toe_1_col", "left_toe_1_col",]

        # --- Make all geoms have contype and conaffinity of 0 ---
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        # --- Define specific contact pairs ---
        for g_name in foot_geoms:
            spec.add_pair(geomname1="floor", geomname2=g_name)

        return spec
