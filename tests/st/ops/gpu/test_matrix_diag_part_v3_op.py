# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore import dtype as mstype
from mindspore.ops.operations.array_ops import MatrixDiagPartV3


class MatrixDiagPartV3Net(nn.Cell):
    def __init__(self, align='LEFT_RIGHT'):
        super(MatrixDiagPartV3Net, self).__init__()
        self.matrix_diag_dart_v3 = MatrixDiagPartV3(align=align)

    def construct(self, x, k, padding_value):
        return self.matrix_diag_dart_v3(x, k, padding_value)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_matrix_diag_part_v3():
    """
    Feature: MatrixDiagPartV3 operator.
    Description: Compatible with np.diag.
    Expectation: The result matches numpy.
    """
    context.set_context(device_target="GPU")
    align = 'RIGHT_LEFT'
    input_x = Tensor(np.array([[[1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [9, 8, 7, 6]],
                               [[5, 4, 3, 2],
                                [1, 2, 3, 4],
                                [5, 6, 7, 8]]]), mstype.float32)
    k = Tensor(np.array([1, 3]), mstype.int32)
    padding_value = Tensor(np.array(9), mstype.float32)

    result = MatrixDiagPartV3(align=align)(input_x, k, padding_value).asnumpy()

    expect = np.array([[[9, 9, 4],
                        [9, 3, 8],
                        [2, 7, 6]],
                       [[9, 9, 2],
                        [9, 3, 4],
                        [4, 3, 8]]], np.float32)

    np.testing.assert_allclose(result, expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_matrix_diag_part_v3_1():
    """
    Feature: MatrixDiagPartV3 operator.
    Description: Compatible with np.diag.
    Expectation: The result matches numpy.
    """
    context.set_context(device_target="GPU")
    align = 'RIGHT_LEFT'
    input_x = Tensor(np.array([[[1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [9, 8, 7, 6]],
                               [[5, 4, 3, 2],
                                [1, 2, 3, 4],
                                [5, 6, 7, 8]]]), mstype.float32)

    k = Tensor(1, mstype.int32)
    padding_value = Tensor(0, mstype.float32)
    result = MatrixDiagPartV3(align=align)(input_x, k, padding_value).asnumpy()
    expect = np.array([[2, 7, 6],
                       [4, 3, 8]], np.float32)
    np.testing.assert_allclose(result, expect)

    k = Tensor((-1, 2), mstype.int32)
    padding_value = Tensor(0, mstype.float32)
    result = MatrixDiagPartV3(align=align)(input_x, k, padding_value).asnumpy()
    expect = np.array([[[0, 3, 8],
                        [2, 7, 6],
                        [1, 6, 7],
                        [5, 8, 0]],
                       [[0, 3, 4],
                        [4, 3, 8],
                        [5, 2, 7],
                        [1, 6, 0]]], np.float32)
    np.testing.assert_allclose(result, expect)

    align = "LEFT_RIGHT"
    k = Tensor((-1, 2), mstype.int32)
    padding_value = Tensor(0, mstype.float32)
    result = MatrixDiagPartV3(align=align)(input_x, k, padding_value).asnumpy()
    expect = np.array([[[3, 8, 0],
                        [2, 7, 6],
                        [1, 6, 7],
                        [0, 5, 8]],
                       [[3, 4, 0],
                        [4, 3, 8],
                        [5, 2, 7],
                        [0, 1, 6]]], np.float32)
    np.testing.assert_allclose(result, expect)

    align = 'RIGHT_LEFT'
    k = Tensor((-2, -1), mstype.int32)
    padding_value = Tensor(0, mstype.float32)
    result = MatrixDiagPartV3(align=align)(input_x, k, padding_value).asnumpy()
    expect = np.array([[[5, 8],
                        [9, 0]],
                       [[1, 6],
                        [5, 0]]], np.float32)
    np.testing.assert_allclose(result, expect)

    k = Tensor((1, 3), mstype.int32)
    padding_value = Tensor(9, mstype.float32)
    result = MatrixDiagPartV3(align=align)(input_x, k, padding_value).asnumpy()
    expect = np.array([[[9, 9, 4],
                        [9, 3, 8],
                        [2, 7, 6]],
                       [[9, 9, 2],
                        [9, 3, 4],
                        [4, 3, 8]]], np.float32)
    np.testing.assert_allclose(result, expect)
