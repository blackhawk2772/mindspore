/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <set>
#include "ops/scatter_nd.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
constexpr int64_t kScatterNdInputNum = 2LL;
void ScatterNdCheckShape(const PrimitivePtr &prim, const AbstractBasePtrList &inputs, const ShapeVector &out_shape) {
  auto indices_shape_ptr = inputs[kInputIndex0]->BuildShape();
  ShapeVector indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(indices_shape_ptr)[kShape];
  const int64_t kIndicesRank = 2LL;
  (void)CheckAndConvertUtils::CheckInteger("rank(indices)", SizeToLong(indices_shape.size()), kGreaterEqual,
                                           kIndicesRank, prim->name());
  auto updates_shape_ptr = inputs[kInputIndex1]->BuildShape();
  ShapeVector updates_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(updates_shape_ptr)[kShape];

  if (out_shape.empty()) {
    MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', the input 'shape' can not be empty.";
  }

  // the last dimension of indices_shape, use the same variable 'N' as document.
  if (indices_shape.back() == abstract::Shape::SHP_ANY) {
    MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', the last dimension of 'indices' can not be dynamic.";
  }
  size_t n = LongToSize(indices_shape.back());
  if (n > out_shape.size()) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim->name()
      << "', if the rank of output tensor is 'P' (length of the 'shape'), and the last dimension of 'indices' is "
      << "'N', the 'N' should be less than or equal to 'P', but got P = " << out_shape.size() << ", N = " << n;
  }
  // the rank of updates is Q-1+P-N
  if (updates_shape.size() != indices_shape.size() - 1 + out_shape.size() - n) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim->name() << "', if the rank of 'indices' is 'Q', the rank of 'updates' is 'R', "
      << "the rank of output tensor is 'P' (length of the 'shape'), and the last dimension of 'indices' is 'N', "
      << "then 'R' should be equal to 'Q - 1 + P - N'. but got 'R' = " << updates_shape.size()
      << ", 'Q' = " << indices_shape.size() << ", 'P' = " << out_shape.size() << ", 'N' = " << n;
  }

  // updates.shape = indices.shape[:-1] + shape[indices.shape[-1]:]
  bool constrain = true;
  for (size_t i = 0; i + 1 < indices_shape.size(); ++i) {
    auto is_dyn = ((updates_shape[i] == abstract::Shape::SHP_ANY) || (indices_shape[i] == abstract::Shape::SHP_ANY));
    if ((updates_shape[i] != indices_shape[i]) && (!is_dyn)) {
      constrain = false;
      break;
    }
  }
  size_t si = n;
  size_t ui = indices_shape.size() - 1;
  for (; si < out_shape.size(); ++si, ++ui) {
    auto is_dyn = ((updates_shape[ui] == abstract::Shape::SHP_ANY) || (out_shape[si] == abstract::Shape::SHP_ANY));
    if ((updates_shape[ui] != out_shape[si]) && (!is_dyn)) {
      constrain = false;
      break;
    }
  }
  if (!constrain) {
    std::ostringstream buffer;
    buffer << "For '" << prim->name()
           << "', if the last dimension of 'indices' is 'N', the shape of 'updates' should be the concatenation of "
           << "'indices.shape[:-1]' and 'shape[N:]'. but got 'indices.shape' is " << indices_shape_ptr->ToString()
           << ", 'updates.shape' is " << updates_shape_ptr->ToString() << ", 'shape' is (";
    for (auto item : out_shape) {
      buffer << item << ", ";
    }
    buffer << ").";
    MS_EXCEPTION(ValueError) << buffer.str();
  }
}

TypePtr ScatterNdInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto dtype = input_args[kInputIndex1]->BuildType();
  (void)CheckAndConvertUtils::CheckSubClass("updates", dtype, {kTensorType}, prim->name());
  return dtype;
}

abstract::ShapePtr ExtrectScatterNdShape(const PrimitivePtr &prim, const AbstractBasePtrList &inputs, bool *is_dyn) {
  auto prim_name = prim->name();
  ShapeVector out_shape;
  *is_dyn = false;
  if (inputs.size() > static_cast<size_t>(kScatterNdInputNum)) {
    auto shape = inputs[kInputIndex2];
    auto shape_value = shape->BuildValue();
    MS_EXCEPTION_IF_NULL(shape_value);
    if (shape->isa<abstract::AbstractTuple>()) {
      out_shape = CheckAndConvertUtils::CheckTupleInt("input[shape]", shape_value, prim_name);
    } else if (shape->isa<abstract::AbstractTensor>()) {
      if (shape_value->isa<tensor::Tensor>()) {
        out_shape = CheckAndConvertUtils::CheckTensorIntValue("shape", shape_value, prim_name);
      } else if (shape_value->isa<ValueTuple>()) {
        out_shape = CheckAndConvertUtils::CheckTupleInt("input[shape]", shape_value, prim_name);
      } else {
        *is_dyn = true;
        auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, inputs, kScatterNdInputNum);
        MS_EXCEPTION_IF_NULL(shape_ptr);
        auto shape_shape = shape_ptr->shape();
        if (shape_shape.size() != 1) {
          MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the input [shape] must be 1-D Tensor, but got "
                                   << shape_shape.size() << "-D.";
        }

        auto shape_len = LongToSize(shape_shape[0]);
        auto abs_shape = shape->cast<abstract::AbstractTensorPtr>();
        MS_EXCEPTION_IF_NULL(abs_shape);

        auto shape_min_value = abs_shape->get_min_value();
        auto shape_max_value = abs_shape->get_max_value();
        if (shape_min_value == nullptr || shape_max_value == nullptr) {
          for (size_t i = 0; i < shape_len; i++) {
            out_shape.push_back(abstract::Shape::SHP_ANY);
          }
          return std::make_shared<abstract::Shape>(out_shape);
        }

        auto min_shape = GetValue<std::vector<int64_t>>(shape_min_value);
        auto max_shape = GetValue<std::vector<int64_t>>(shape_max_value);
        if (min_shape.size() != shape_len || max_shape.size() != shape_len) {
          MS_LOG(EXCEPTION)
            << "For '" << prim_name
            << "', shape's min and max value must has lengths equal to shape itself, but got min value len: "
            << min_shape.size() << ", max value len: " << max_shape.size() << ", shape len: " << shape_len << ".";
        }

        for (size_t i = 0; i < shape_len; i++) {
          if (min_shape[i] == max_shape[i]) {
            out_shape.push_back(min_shape[i]);
          } else {
            out_shape.push_back(abstract::Shape::SHP_ANY);
          }
        }
        return std::make_shared<abstract::Shape>(out_shape, min_shape, max_shape);
      }
    }
  } else {
    auto shape_attr = prim->GetAttr("shape");
    MS_EXCEPTION_IF_NULL(shape_attr);
    out_shape = GetValue<ShapeVector>(shape_attr);
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

abstract::BaseShapePtr ScatterNdInferShape(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  bool is_dyn_output;
  auto shape = ExtrectScatterNdShape(prim, input_args, &is_dyn_output);
  const auto &out_shape = shape->shape();
  if (!std::all_of(out_shape.begin(), out_shape.end(), [is_dyn_output](int64_t item) {
        return item >= 1 || (is_dyn_output && (item == abstract::Shape::SHP_ANY));
      })) {
    std::ostringstream buffer;
    buffer << "For 'ScatterNd', the input[shape] should be a tuple with all positive item. but got (";
    for (auto item : out_shape) {
      buffer << item << ", ";
    }
    buffer << ").";
    MS_EXCEPTION(ValueError) << buffer.str();
  }
  ScatterNdCheckShape(prim, input_args, out_shape);
  return shape;
}

AbstractBasePtr ScatterNdInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  const std::set<TypePtr> valid_indices_types = {kInt32, kInt64};
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kScatterNdInputNum, name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", input_args[0]->BuildType(), valid_indices_types, name);
  if (input_args.size() > static_cast<size_t>(kScatterNdInputNum)) {
    auto shape_type = input_args[kInputIndex2]->BuildType();
    if (!shape_type->isa<TensorType>()) {
      (void)CheckAndConvertUtils::CheckTypeValid("shape", shape_type, {kTuple}, name);
    }
  }

  auto infer_type = ScatterNdInferType(primitive, input_args);
  auto infer_shape = ScatterNdInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(ScatterNd, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(ScatterNd, prim::kPrimScatterNd, ScatterNdInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
