/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_SPACETOTENSORRT_PLUGIN_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_SPACETOTENSORRT_PLUGIN_H_

#include <string>
#include <vector>
#include "src/extendrt/delegate/tensorrt/op/tensorrt_op.h"
#include "src/extendrt/delegate/tensorrt/op/tensorrt_plugin.h"

namespace mindspore::lite {
class SpaceToBatchTensorRT : public TensorRTOp {
 public:
  SpaceToBatchTensorRT(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                       const std::vector<TensorInfo> &out_tensors, std::string name)
      : TensorRTOp(base_operator, in_tensors, out_tensors, name) {}

  ~SpaceToBatchTensorRT() override = default;

  int AddInnerOp(TensorRTContext *ctx) override;

  int IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                const std::vector<TensorInfo> &out_tensors) override;
};

constexpr auto SPACETOTENSORRT_PLUGIN_NAME{"SpaceToBatchPlugin"};
class SpaceToBatchPlugin : public TensorRTPlugin {
 public:
  SpaceToBatchPlugin(const std::string name, int in, int ic, int ih, int iw, int bh, int ph0, int ph1, int pw0, int pw1,
                     uint32_t device_id)
      : TensorRTPlugin(name, std::string(SPACETOTENSORRT_PLUGIN_NAME), device_id),
        in_(in),
        ic_(ic),
        ih_(ih),
        iw_(iw),
        bh_(bh),
        ph0_(ph0),
        ph1_(ph1),
        pw0_(pw0),
        pw1_(pw1) {}

  SpaceToBatchPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc)
      : TensorRTPlugin(std::string(name), std::string(SPACETOTENSORRT_PLUGIN_NAME)) {
    const nvinfer1::PluginField *fields = fc->fields;
    in_ = static_cast<const int *>(fields[0].data)[0];
    ic_ = static_cast<const int *>(fields[1].data)[0];
    ih_ = static_cast<const int *>(fields[2].data)[0];
    iw_ = static_cast<const int *>(fields[3].data)[0];
    bh_ = static_cast<const int *>(fields[4].data)[0];
    ph0_ = static_cast<const int *>(fields[5].data)[0];
    ph1_ = static_cast<const int *>(fields[6].data)[0];
    pw0_ = static_cast<const int *>(fields[7].data)[0];
    pw1_ = static_cast<const int *>(fields[8].data)[0];
  }

  SpaceToBatchPlugin(const char *name, const void *serialData, size_t serialLength)
      : TensorRTPlugin(std::string(name), std::string(SPACETOTENSORRT_PLUGIN_NAME)) {
    DeserializeValue(&serialData, &serialLength, &in_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &ic_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &ih_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &iw_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &bh_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &ph0_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &ph1_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &pw0_, sizeof(int));
    DeserializeValue(&serialData, &serialLength, &pw1_, sizeof(int));
  }

  SpaceToBatchPlugin() = delete;

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs *inputs, int nbInputDims,
                                          nvinfer1::IExprBuilder &exprBuilder) noexcept override;

 private:
  int RunCudaSpaceToBatch(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs, void *const *outputs,
                          cudaStream_t stream);
  size_t in_;
  size_t ic_;
  size_t ih_;
  size_t iw_;
  size_t bh_;
  size_t ph0_;
  size_t ph1_;
  size_t pw0_;
  size_t pw1_;
  const std::string layer_name_;
  std::string name_space_;
};
class SpaceToBatchPluginCreater : public TensorRTPluginCreater<SpaceToBatchPlugin> {
 public:
  SpaceToBatchPluginCreater() : TensorRTPluginCreater(std::string(SPACETOTENSORRT_PLUGIN_NAME)) {}
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_OP_SPACETOTENSORRT_PLUGIN_H_
