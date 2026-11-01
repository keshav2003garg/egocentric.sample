"""Layers in logging order. Each one logs its data if the clip has its files, and supplies a side-panel tab."""
from ego_rerun.layers import camera, contact, depth, hands, masks, steps

LAYERS = [camera, depth, masks, hands, contact, steps]
TAB_ORDER = [steps, contact, hands, masks, depth, camera]
