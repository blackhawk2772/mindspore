/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <vector>
#include <map>
#include <set>
#include <string>

#include "ops/zeta.h"
#include "ops/op_utils.h"
#include "utils/tensor_construct_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ZetaInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape_ptr = CheckAndConvertUtils::GetTensorInputShape("Zeta", input_args, 0);
  auto q_shape_ptr = CheckAndConvertUtils::GetTensorInputShape("Zeta", input_args, 1);
  auto x_shape = x_shape_ptr->shape();
  auto q_shape = q_shape_ptr->shape();
  CheckAndConvertUtils::Check("input_x size", int64_t(x_shape.size()), kGreaterEqual, int64_t(q_shape.size()),
                              prim_name);
  if (x_shape.size() != 0 && x_shape[0] == 0) {
    MS_EXCEPTION(ValueError) << "For Zeta, the input_x must have value.";
  }
  if (q_shape.size() != 0 && q_shape[0] == 0) {
    MS_EXCEPTION(ValueError) << "For Zeta, the input_q must have value.";
  }
  if (*x_shape_ptr != *q_shape_ptr) {
    MS_EXCEPTION(ValueError) << primitive->name() << "Shape of x" << x_shape_ptr->ToString()
                             << " are not consistent with the shape q" << q_shape_ptr->ToString();
  }
  return x_shape_ptr;
}
TypePtr ZetaInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  auto input_x = input_args[0]->BuildType();
  auto input_q = input_args[1]->BuildType();
  std::map<std::string, TypePtr> args_type;
  (void)args_type.insert({"x", input_x});
  (void)args_type.insert({"q", input_q});
  auto output_type = CheckAndConvertUtils::CheckTensorTypeSame(args_type, valid_types, primitive->name());
  return output_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Zeta, BaseOperator);
AbstractBasePtr ZetaInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = ZetaInferType(primitive, input_args);
  auto infer_shape = ZetaInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(Zeta, prim::kPrimZeta, ZetaInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
