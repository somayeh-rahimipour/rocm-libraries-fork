#!/usr/bin/env python3

# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import List, Optional, OrderedDict, Callable
import sys
import os
import warnings

sys.path.append(f"{os.path.dirname(__file__)}/../")

from tuner.base_tuner import BaseTuner, TunerArgs

# batch_memcpy intentionally has no thread-block-dimension tune param (see
# _get_grid_div_x), so kernel_tuner warns "None of the tunable parameters specify
# thread block dimensions!" on every tune_kernel call. That is expected here, so
# silence just that one message (matched by its text prefix) rather than muting
# all UserWarnings.
warnings.filterwarnings(
    "ignore",
    message="None of the tunable parameters specify thread block dimensions",
    category=UserWarning,
)

"""
Inclusive range for params tuning, edit these to adjust tuning grid range.
"""
BLOCK_SIZES = list(range(64, 1024 + 1, 64))
ITEMS_PER_THREAD = [1, 2, 4, 8, 16, 32]
WLEV_THRESHOLDS = [64, 128, 256, 512]
BLEV_THRESHOLDS = [1024, 2048, 4096, 8192]

# Mixed buffer distribution (tlev + wlev + blev together) so a single tuning run
# exercises all three copy levels and yields one complete batch_memcpy_config per
# item size -- the "one config per type" shape confgen expects.
MIXED_BUFFERS = (1000, 1000, 1000)  # (num_tlev, num_wlev, num_blev)

# The library selects a config at dispatch purely by sizeof(value_type)
# (batch_memcpy_config_selector), so we tune one representative naturally-aligned
# type per size bucket and label the run with a real C++ type name of that size.
# confgen then buckets by sizeof exactly like every other algo -- no confgen-side
# special casing needed. Alignment is intentionally collapsed: the selector keys
# on size only, and one config serves every alignment of a given size.
ITEM_SIZE_TYPES = {
    1: "int8_t",
    2: "short",
    4: "int",
    8: "int64_t",
}


class Tuner(BaseTuner):
    @classmethod
    def _get_default_args(cls) -> TunerArgs:
        return TunerArgs(algo_full_name='device_batch_memcpy')

    def __init__(self, args: TunerArgs) -> None:
        super().__init__(args)
        self._scenario: Optional[dict] = None

    def _get_tune_params(self) -> OrderedDict:
        assert self._scenario is not None, "tune_all() must set the scenario first"
        params: OrderedDict = OrderedDict()

        params['TUNE_NON_BLEV_BLOCK_SIZE'] = BLOCK_SIZES
        params['TUNE_NON_BLEV_IPT']        = ITEMS_PER_THREAD
        params['TUNE_TLEV_IPT']            = ITEMS_PER_THREAD
        params['TUNE_BLEV_BLOCK_SIZE']     = BLOCK_SIZES
        params['TUNE_BLEV_IPT']            = ITEMS_PER_THREAD
        params['TUNE_WLEV_THRESHOLD']      = WLEV_THRESHOLDS
        params['TUNE_BLEV_THRESHOLD']      = BLEV_THRESHOLDS

        s = self._scenario
        params['TUNE_ITEM_SIZE']  = [s['item_size']]
        params['TUNE_ITEM_ALIGN'] = [s['item_align']]
        params['TUNE_NUM_TLEV']   = [s['num_tlev']]
        params['TUNE_NUM_WLEV']   = [s['num_wlev']]
        params['TUNE_NUM_BLEV']   = [s['num_blev']]
        return params

    def _get_grid_div_x(self) -> List[str]:
        return []

    def _get_problem_size(self, key_type: str, value_type: Optional[str] = None):
        return self.bytes_size

    def _get_restrictions(
        self, key_type: str, value_type: Optional[str] = None
    ) -> Callable[[dict], bool]:
        # batch_memcpy restrictions do not depend on a key type.
        def validate(params):
            nb_bs   = params['TUNE_NON_BLEV_BLOCK_SIZE']
            nb_ipt  = params['TUNE_NON_BLEV_IPT']
            tl_ipt  = params['TUNE_TLEV_IPT']
            bl_bs   = params['TUNE_BLEV_BLOCK_SIZE']
            bl_ipt  = params['TUNE_BLEV_IPT']
            wlev_th = params['TUNE_WLEV_THRESHOLD']
            blev_th = params['TUNE_BLEV_THRESHOLD']
            item_size = params['TUNE_ITEM_SIZE']

            # Block sizes must be multiples of 64 within [64, 1024].
            if not (64 <= nb_bs <= 1024 and nb_bs % 64 == 0):
                return False
            if not (64 <= bl_bs <= 1024 and bl_bs % 64 == 0):
                return False

            # Size-class ordering: warp threshold strictly below block threshold.
            if wlev_th >= blev_th:
                return False

            # LDS/tile-size sanity: per-block tile bytes bounded by 64 KiB.
            if nb_bs * nb_ipt * item_size > 65536:
                return False
            if bl_bs * bl_ipt * item_size > 65536:
                return False

            # High items_per_thread on gfx1030 cause HSA_STATUS_ERROR_INVALID_ISA.
            if params.get("arch_name") == "gfx1030" and max(nb_ipt, tl_ipt, bl_ipt) > 28:
                return False

            return True

        return validate

    def tune_all(self) -> None:
        """Tune one config per item-size bucket using the mixed buffer distribution."""
        num_tlev, num_wlev, num_blev = MIXED_BUFFERS
        for item_size, type_name in ITEM_SIZE_TYPES.items():
            self._scenario = {
                'item_size': item_size,
                'item_align': item_size,  # naturally aligned -> sizeof == item_size
                'num_tlev': num_tlev,
                'num_wlev': num_wlev,
                'num_blev': num_blev,
            }
            # key_type is a real C++ type of the same size, so confgen buckets the
            # result by sizeof(value_type) like every other algo, and cache/output
            # file names stay unique per size.
            self.tune_type(type_name)


if __name__ == "__main__":
    Tuner.cli()
