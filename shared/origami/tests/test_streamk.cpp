/*******************************************************************************
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

#include <catch2/catch_test_macros.hpp>
#include "common.hpp"

namespace {

inline origami::problem_t make_problem_with_tile_count(
    size_t mt_m, size_t mt_n, size_t tiles, size_t batch = 1) {
  return make_problem(mt_m, mt_n * tiles, 64,
                      origami::transpose_t::T, origami::transpose_t::N,
                      batch);
}

}  // namespace

TEST_CASE("Origami streamk: select_hybrid_mode without a genuine CU cap stays static",
          "[origami][streamk][hybrid]") {
  auto config  = make_config(128, 128, 32, 16, 16, 16, false, 1, 1);
  auto problem = make_problem_with_tile_count(128, 128, 4096);
  for (int arch : {942, 950}) {
    auto hardware = make_hardware(arch);
    REQUIRE(origami::streamk::select_hybrid_mode(problem, hardware, config, 0)
            == origami::hybrid_mode_t::static_);
    REQUIRE(origami::streamk::select_hybrid_mode(problem, hardware, config, hardware.N_CU)
            == origami::hybrid_mode_t::static_);
  }
}

TEST_CASE("Origami streamk: select_hybrid_mode sm_count_target=0 uses N_CU",
          "[origami][streamk][hybrid]") {
  auto config  = make_config(128, 128, 32);
  auto problem = make_problem(4096, 4096, 64);
  for (int arch : {942, 950}) {
    auto hardware = make_hardware(arch);
    auto a = origami::streamk::select_hybrid_mode(problem, hardware, config, 0);
    auto b = origami::streamk::select_hybrid_mode(problem, hardware, config, hardware.N_CU);
    REQUIRE(a == b);
  }
}

TEST_CASE("Origami streamk: gfx942 tree gates on the grid_waves threshold",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(942);
  auto config   = make_config(128, 128, 32, 16, 16, 16, false, 1, 2);
  auto problem  = make_problem(8192, 8192, 64);

  REQUIRE(origami::streamk::gfx942_values::select_hybrid_mode(problem, hardware, config, 0)
          == origami::hybrid_mode_t::static_);
  REQUIRE(origami::streamk::gfx942_values::select_hybrid_mode(problem, hardware, config, 64)
          == origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: gfx950 tree gates each leaf on its threshold",
          "[origami][streamk][hybrid]") {
  struct gate_case {
    const char*            leaf;
    origami::hybrid_mode_t expected;
    origami::problem_t     problem;
    origami::config_t      config;
    size_t                 smt;
  };

  auto config = [](size_t mt_m, size_t mt_n, size_t mt_k, int occ) {
    return make_config(mt_m, mt_n, mt_k, 16, 16, 16, false, 1, occ);
  };

  const gate_case cases[] = {
      {"grid_efficiency<=0.23, min_mn<=1088, m_dim<=3277 -> static", origami::hybrid_mode_t::static_,
       make_problem(128, 128, 65536), config(128, 128, 32, 3), 128},
      {"grid_efficiency<=0.23, min_mn<=1088, m_dim>3277 -> dynamic", origami::hybrid_mode_t::dynamic,
       make_problem(4096, 128, 65536), config(256, 128, 32, 3), 128},
      {"grid_efficiency<=0.23, min_mn>1088, tiles<=34 -> dynamic", origami::hybrid_mode_t::dynamic,
       make_problem(2048, 2048, 65536), config(512, 512, 32, 3), 128},
      {"grid_efficiency<=0.23, min_mn>1088, tiles>34 -> static", origami::hybrid_mode_t::static_,
       make_problem(3072, 3072, 65536), config(512, 512, 32, 3), 240},
      {"grid_efficiency>0.23, tiles_per_cu<=0.29, static_skgrid<=68 -> dynamic", origami::hybrid_mode_t::dynamic,
       make_problem(256, 256, 64), config(128, 128, 32, 3), 128},
      {"grid_efficiency>0.23, tiles_per_cu<=0.29, static_skgrid>68, iters_per_tile<=458 -> static", origami::hybrid_mode_t::static_,
       make_problem(128, 9344, 64), config(128, 128, 64, 3), 255},
      {"grid_efficiency>0.23, tiles_per_cu<=0.29, static_skgrid>68, iters_per_tile>458 -> dynamic", origami::hybrid_mode_t::dynamic,
       make_problem(128, 9344, 65536), config(128, 128, 64, 3), 255},
      {"grid_efficiency>0.23, tiles_per_cu>0.29, active_cus<=240 -> dynamic", origami::hybrid_mode_t::dynamic,
       make_problem(8192, 8192, 8192), config(128, 128, 32, 3), 128},
      {"grid_efficiency>0.23, tiles_per_cu>0.29, active_cus>240, occupancy<=2.5 -> dynamic", origami::hybrid_mode_t::dynamic,
       make_problem(16384, 16384, 64), config(128, 128, 32, 2), 250},
      {"grid_efficiency>0.23, tiles_per_cu>0.29, active_cus>240, occupancy>2.5 -> static", origami::hybrid_mode_t::static_,
       make_problem(16384, 16384, 64), config(128, 128, 32, 3), 250},
  };

  auto hardware = make_hardware(950);
  for (auto const& gc : cases) {
    CAPTURE(gc.leaf);
    REQUIRE(origami::streamk::gfx950_values::select_hybrid_mode(gc.problem, hardware, gc.config, gc.smt)
            == gc.expected);
  }
}
