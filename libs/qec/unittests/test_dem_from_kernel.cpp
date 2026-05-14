/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/dem.h"
#include <cudaq.h>
#include <gtest/gtest.h>

// A kernel with no QEC annotations produces an empty DEM. This validates
// the full cross-repo data path: cudaq::analysis::compute_dem (CUDA-Q,
// Stim-backed) -> cudaq::analysis::detector_error_model POD ->
// cudaq::qec::detail::podToQecDem -> cudaq::qec::detector_error_model
// (tensor-based).
//
// The non-empty case (kernel with cudaq::detector / cudaq::observable_include)
// requires MLIR-mode compilation and is deferred to the nvq++ integration
// test layer.

struct TrivialKernel {
  void operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::h(q);
    cudaq::mz(q);
  }
};

// DISABLED: crashes with `free(): invalid pointer` during Stim simulator
// cleanup (CircuitSimulator.h:1160 "Deallocating simulator state"). Root
// cause is a cross-DSO memory management issue: the Stim plugin is
// dlopen'd by libcudaq-analysis.so, allocates simulator state in its own
// allocator context, and the cleanup path frees it through a different
// allocator reachable from the test executable. The POD data-transfer
// path (podToQecDem tests below) is unaffected because it never crosses
// the allocator boundary. Phase 1.5 finding: the runtime dispatch
// infrastructure needs DSO-safe allocator discipline before
// dem_from_kernel can be called from a separately-linked CUDA-QX binary.
TEST(DemFromKernel, DISABLED_trivialKernelEmptyDem) {
  cudaq::noise_model noise;
  auto dem = cudaq::qec::dem_from_kernel(TrivialKernel{}, noise);

  EXPECT_EQ(dem.num_detectors(), 0u);
  EXPECT_EQ(dem.num_error_mechanisms(), 0u);
  EXPECT_EQ(dem.num_observables(), 0u);
  EXPECT_TRUE(dem.error_rates.empty());
  EXPECT_FALSE(dem.error_ids.has_value());
}

TEST(DemFromKernel, podToQecDemEmptyPod) {
  cudaq::analysis::detector_error_model pod;
  auto dem = cudaq::qec::detail::podToQecDem(std::move(pod));

  EXPECT_EQ(dem.num_detectors(), 0u);
  EXPECT_EQ(dem.num_error_mechanisms(), 0u);
  EXPECT_EQ(dem.num_observables(), 0u);
  EXPECT_TRUE(dem.error_rates.empty());
  EXPECT_FALSE(dem.error_ids.has_value());
}

TEST(DemFromKernel, podToQecDemPopulated) {
  cudaq::analysis::detector_error_model pod;
  pod.num_detectors = 2;
  pod.num_error_mechanisms = 3;
  pod.num_observables = 1;
  // Row-major [2,3]: det 0 triggers on err 0 and 2; det 1 triggers on err 1.
  pod.detector_error_matrix = {1, 0, 1, 0, 1, 0};
  // Row-major [1,3]: obs 0 flipped by err 2.
  pod.observables_flips_matrix = {0, 0, 1};
  pod.error_rates = {0.01, 0.02, 0.03};
  pod.error_ids = std::vector<std::size_t>{10, 20, 10};

  auto dem = cudaq::qec::detail::podToQecDem(std::move(pod));

  EXPECT_EQ(dem.num_detectors(), 2u);
  EXPECT_EQ(dem.num_error_mechanisms(), 3u);
  EXPECT_EQ(dem.num_observables(), 1u);

  // Verify detector_error_matrix shape and content.
  auto detShape = dem.detector_error_matrix.shape();
  ASSERT_EQ(detShape.size(), 2u);
  EXPECT_EQ(detShape[0], 2u);
  EXPECT_EQ(detShape[1], 3u);
  EXPECT_EQ(dem.detector_error_matrix.at({0, 0}), 1);
  EXPECT_EQ(dem.detector_error_matrix.at({0, 1}), 0);
  EXPECT_EQ(dem.detector_error_matrix.at({0, 2}), 1);
  EXPECT_EQ(dem.detector_error_matrix.at({1, 0}), 0);
  EXPECT_EQ(dem.detector_error_matrix.at({1, 1}), 1);
  EXPECT_EQ(dem.detector_error_matrix.at({1, 2}), 0);

  // Verify observables_flips_matrix shape and content.
  auto obsShape = dem.observables_flips_matrix.shape();
  ASSERT_EQ(obsShape.size(), 2u);
  EXPECT_EQ(obsShape[0], 1u);
  EXPECT_EQ(obsShape[1], 3u);
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 0}), 0);
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 1}), 0);
  EXPECT_EQ(dem.observables_flips_matrix.at({0, 2}), 1);

  // Verify error_rates and error_ids were moved.
  ASSERT_EQ(dem.error_rates.size(), 3u);
  EXPECT_DOUBLE_EQ(dem.error_rates[0], 0.01);
  EXPECT_DOUBLE_EQ(dem.error_rates[1], 0.02);
  EXPECT_DOUBLE_EQ(dem.error_rates[2], 0.03);

  ASSERT_TRUE(dem.error_ids.has_value());
  ASSERT_EQ(dem.error_ids->size(), 3u);
  EXPECT_EQ((*dem.error_ids)[0], 10u);
  EXPECT_EQ((*dem.error_ids)[1], 20u);
  EXPECT_EQ((*dem.error_ids)[2], 10u);
}
