/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/dem.h"

namespace cudaq::qec::detail {

detector_error_model
podToQecDem(cudaq::analysis::detector_error_model &&pod) {
  detector_error_model result;
  const auto D = pod.num_detectors;
  const auto E = pod.num_error_mechanisms;
  const auto K = pod.num_observables;

  if (D > 0 && E > 0) {
    result.detector_error_matrix = cudaqx::tensor<uint8_t>({D, E});
    result.detector_error_matrix.copy(pod.detector_error_matrix.data());
  }

  if (K > 0 && E > 0) {
    result.observables_flips_matrix = cudaqx::tensor<uint8_t>({K, E});
    result.observables_flips_matrix.copy(
        pod.observables_flips_matrix.data());
  }

  result.error_rates = std::move(pod.error_rates);
  result.error_ids = std::move(pod.error_ids);

  return result;
}

} // namespace cudaq::qec::detail
