/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPACE_TO_BATCH_ND_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPACE_TO_BATCH_ND_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SpaceToBatchNDCpuKernelMod : public NativeCpuKernelMod {
 public:
  SpaceToBatchNDCpuKernelMod() = default;
  ~SpaceToBatchNDCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
              const std::vector<KernelTensorPtr> &outputs,
              const std::map<uint32_t, tensor::TensorPtr> &others = std::map<uint32_t, tensor::TensorPtr>()) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 private:
  void CheckParam();

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  using SpaceToBatchNDFunc =
    std::function<bool(SpaceToBatchNDCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, SpaceToBatchNDFunc>> func_list_;
  SpaceToBatchNDFunc kernel_func_;

  std::vector<std::vector<int64_t>> paddings_;
  std::vector<int64_t> block_size_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  size_t block_rank_;
  size_t off_set_;
  int64_t input_size_;
  int64_t output_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPACE_TO_BATCH_ND_CPU_KERNEL_H_