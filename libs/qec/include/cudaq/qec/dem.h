/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/analysis/dem.h"
#include "cudaq/qec/detector_error_model.h"
#include <utility>

namespace cudaq::qec {

namespace detail {
/// Convert the flat POD returned by cudaq::analysis::compute_dem into the
/// tensor-based cudaq::qec::detector_error_model.
detector_error_model
podToQecDem(cudaq::analysis::detector_error_model &&pod);
} // namespace detail

/// @brief Compute a detector error model by running a CUDA-Q kernel under
/// Stim-backed DEM analysis and converting the result to the CUDA-QX tensor
/// representation.
///
/// Delegates to `cudaq::analysis::compute_dem`, which runs the kernel under
/// an internal Stim simulator with `nvqir::dem_analysis::make_scope`. The
/// flat POD returned by the CUDA-Q analysis primitive is converted to the
/// tensor-based `cudaq::qec::detector_error_model` via `detail::podToQecDem`.
///
/// @param kernel  Any callable invocable with @p args (CUDA-Q kernel).
/// @param noise   Noise model applied during analysis.
/// @param args    Arguments forwarded to the kernel invocation.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
detector_error_model dem_from_kernel(QuantumKernel &&kernel,
                                     cudaq::noise_model &noise,
                                     Args &&...args) {
  auto pod = cudaq::analysis::compute_dem(
      std::forward<QuantumKernel>(kernel), &noise,
      std::forward<Args>(args)...);
  return detail::podToQecDem(std::move(pod));
}

} // namespace cudaq::qec
