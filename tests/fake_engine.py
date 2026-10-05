"""A deliberately misbehaving engine, used to check that verdicts are assigned
correctly. Each route fails in one specific way."""

import logging
import warnings

import numpy as np

NAME = "fake"
DISPLAY = "Fake"
HOMEPAGE = "https://example.invalid"
ROUTES = {"exact": "", "off-by-1pc": "", "warns": "", "logs": "", "raises": "",
          "nan": "", "late-raise": ""}
FAMILIES = {r: {"chain", "near_singular"} for r in ROUTES}
NOT_MODELLED: dict = {}


def version():
    return "0"


def build(route, case):
    ref = case.reference

    def accel(state):
        if route == "raises":
            raise RuntimeError("refused")
        a = ref.accel(state)
        if route == "exact":
            return a
        if route == "nan":
            return a * np.nan
        if route == "warns":
            warnings.warn("mass matrix is badly conditioned", RuntimeWarning)
            return a * 1.01
        if route == "logs":
            logging.getLogger("fake").warning("could not solve")
            return a * 1.01
        if route == "late-raise":
            if state[0] > 0:
                raise RuntimeError("refused part-way")
            return a * 1.01
        return a * 1.01
    return accel
