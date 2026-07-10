#!/usr/bin/env python3

import numpy as np


def relu_backward(input_value, output_gradient):
    return np.where(input_value > 0.0, output_gradient, 0.0)


def conv2d(input_value, weight, bias, padding, strides, dilations):
    batch, input_channels, input_height, input_width = input_value.shape
    output_channels, _, kernel_height, kernel_width = weight.shape
    effective_height = (kernel_height - 1) * dilations[0] + 1
    effective_width = (kernel_width - 1) * dilations[1] + 1
    output_height = (
        input_height + padding[0] + padding[1] - effective_height
    ) // strides[0] + 1
    output_width = (
        input_width + padding[2] + padding[3] - effective_width
    ) // strides[1] + 1
    output = np.zeros(
        (batch, output_channels, output_height, output_width), dtype=np.float64
    )

    for n in range(batch):
        for output_channel in range(output_channels):
            for output_height_index in range(output_height):
                for output_width_index in range(output_width):
                    value = bias[output_channel]
                    for input_channel in range(input_channels):
                        for kernel_height_index in range(kernel_height):
                            for kernel_width_index in range(kernel_width):
                                input_height_index = (
                                    output_height_index * strides[0]
                                    + kernel_height_index * dilations[0]
                                    - padding[0]
                                )
                                input_width_index = (
                                    output_width_index * strides[1]
                                    + kernel_width_index * dilations[1]
                                    - padding[2]
                                )
                                if (
                                    0 <= input_height_index < input_height
                                    and 0 <= input_width_index < input_width
                                ):
                                    value += (
                                        input_value[
                                            n,
                                            input_channel,
                                            input_height_index,
                                            input_width_index,
                                        ]
                                        * weight[
                                            output_channel,
                                            input_channel,
                                            kernel_height_index,
                                            kernel_width_index,
                                        ]
                                    )
                    output[
                        n,
                        output_channel,
                        output_height_index,
                        output_width_index,
                    ] = value
    return output


def conv2d_backward(
    input_value, weight, output_gradient, padding, strides, dilations
):
    input_gradient = np.zeros_like(input_value)
    weight_gradient = np.zeros_like(weight)
    bias_gradient = np.zeros(weight.shape[0], dtype=np.float64)

    for n in range(output_gradient.shape[0]):
        for output_channel in range(output_gradient.shape[1]):
            for output_height_index in range(output_gradient.shape[2]):
                for output_width_index in range(output_gradient.shape[3]):
                    gradient = output_gradient[
                        n,
                        output_channel,
                        output_height_index,
                        output_width_index,
                    ]
                    bias_gradient[output_channel] += gradient
                    for input_channel in range(weight.shape[1]):
                        for kernel_height_index in range(weight.shape[2]):
                            for kernel_width_index in range(weight.shape[3]):
                                input_height_index = (
                                    output_height_index * strides[0]
                                    + kernel_height_index * dilations[0]
                                    - padding[0]
                                )
                                input_width_index = (
                                    output_width_index * strides[1]
                                    + kernel_width_index * dilations[1]
                                    - padding[2]
                                )
                                if (
                                    0 <= input_height_index < input_value.shape[2]
                                    and 0 <= input_width_index < input_value.shape[3]
                                ):
                                    input_gradient[
                                        n,
                                        input_channel,
                                        input_height_index,
                                        input_width_index,
                                    ] += gradient * weight[
                                        output_channel,
                                        input_channel,
                                        kernel_height_index,
                                        kernel_width_index,
                                    ]
                                    weight_gradient[
                                        output_channel,
                                        input_channel,
                                        kernel_height_index,
                                        kernel_width_index,
                                    ] += gradient * input_value[
                                        n,
                                        input_channel,
                                        input_height_index,
                                        input_width_index,
                                    ]
    return input_gradient, weight_gradient, bias_gradient


def max_pool2d(input_value, kernel, padding, strides):
    output_height = (
        input_value.shape[2] + padding[0] + padding[1] - kernel[0]
    ) // strides[0] + 1
    output_width = (
        input_value.shape[3] + padding[2] + padding[3] - kernel[1]
    ) // strides[1] + 1
    output = np.empty(
        (input_value.shape[0], input_value.shape[1], output_height, output_width),
        dtype=np.float64,
    )

    for n in range(output.shape[0]):
        for channel in range(output.shape[1]):
            for output_height_index in range(output_height):
                for output_width_index in range(output_width):
                    best = None
                    for kernel_height_index in range(kernel[0]):
                        for kernel_width_index in range(kernel[1]):
                            input_height_index = (
                                output_height_index * strides[0]
                                + kernel_height_index
                                - padding[0]
                            )
                            input_width_index = (
                                output_width_index * strides[1]
                                + kernel_width_index
                                - padding[2]
                            )
                            if (
                                0 <= input_height_index < input_value.shape[2]
                                and 0 <= input_width_index < input_value.shape[3]
                            ):
                                candidate = input_value[
                                    n,
                                    channel,
                                    input_height_index,
                                    input_width_index,
                                ]
                                if best is None or candidate > best:
                                    best = candidate
                    output[
                        n, channel, output_height_index, output_width_index
                    ] = best
    return output


def max_pool2d_backward(input_value, output_gradient, kernel, padding, strides):
    input_gradient = np.zeros_like(input_value)
    for n in range(output_gradient.shape[0]):
        for channel in range(output_gradient.shape[1]):
            for output_height_index in range(output_gradient.shape[2]):
                for output_width_index in range(output_gradient.shape[3]):
                    best = None
                    best_index = None
                    for kernel_height_index in range(kernel[0]):
                        for kernel_width_index in range(kernel[1]):
                            input_height_index = (
                                output_height_index * strides[0]
                                + kernel_height_index
                                - padding[0]
                            )
                            input_width_index = (
                                output_width_index * strides[1]
                                + kernel_width_index
                                - padding[2]
                            )
                            if (
                                0 <= input_height_index < input_value.shape[2]
                                and 0 <= input_width_index < input_value.shape[3]
                            ):
                                candidate = input_value[
                                    n,
                                    channel,
                                    input_height_index,
                                    input_width_index,
                                ]
                                if best is None or candidate > best:
                                    best = candidate
                                    best_index = (
                                        n,
                                        channel,
                                        input_height_index,
                                        input_width_index,
                                    )
                    input_gradient[best_index] += output_gradient[
                        n, channel, output_height_index, output_width_index
                    ]
    return input_gradient


def finite_difference(value, loss):
    # Central differences independently validate the handwritten references.
    result = np.zeros_like(value)
    flat_value = value.reshape(-1)
    flat_result = result.reshape(-1)
    epsilon = 1.0e-6
    for index in range(flat_value.size):
        original = flat_value[index]
        flat_value[index] = original + epsilon
        positive = loss()
        flat_value[index] = original - epsilon
        negative = loss()
        flat_value[index] = original
        flat_result[index] = (positive - negative) / (2.0 * epsilon)
    return result


def main():
    relu_input = np.array([-1.0, 0.0, 2.0, 3.0])
    relu_output_gradient = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(
        relu_backward(relu_input, relu_output_gradient),
        np.array([0.0, 0.0, 3.0, 4.0]),
    )

    smooth_relu_input = np.array([-2.0, -0.5, 0.5, 2.0])
    smooth_relu_output_gradient = np.array([1.0, 2.0, 3.0, 4.0])
    numerical_relu = finite_difference(
        smooth_relu_input,
        lambda: np.sum(
            np.maximum(smooth_relu_input, 0.0) * smooth_relu_output_gradient
        ),
    )
    np.testing.assert_allclose(
        relu_backward(smooth_relu_input, smooth_relu_output_gradient),
        numerical_relu,
        atol=1.0e-8,
    )

    input_value = np.arange(1.0, 17.0).reshape(1, 1, 4, 4)
    weight = np.array([1.5, -2.0, 0.5, 3.0]).reshape(1, 1, 2, 2)
    bias = np.array([0.25])
    output_gradient = np.array([1.0, -2.0, 0.5, 3.0]).reshape(1, 1, 2, 2)
    padding = (1, 0, 1, 0)
    strides = (2, 2)
    dilations = (2, 1)
    input_gradient, weight_gradient, bias_gradient = conv2d_backward(
        input_value, weight, output_gradient, padding, strides, dilations
    )
    np.testing.assert_array_equal(
        input_gradient,
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
                3.5,
                -12.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.5,
                1.5,
                9.0,
                0.0,
            ]
        ).reshape(1, 1, 4, 4),
    )
    np.testing.assert_array_equal(
        weight_gradient, np.array([18.0, 23.5, 30.0, 42.5]).reshape(1, 1, 2, 2)
    )
    np.testing.assert_array_equal(bias_gradient, np.array([2.5]))

    conv_loss = lambda: np.sum(
        conv2d(input_value, weight, bias, padding, strides, dilations)
        * output_gradient
    )
    np.testing.assert_allclose(
        input_gradient, finite_difference(input_value, conv_loss), atol=1.0e-7
    )
    np.testing.assert_allclose(
        weight_gradient, finite_difference(weight, conv_loss), atol=1.0e-7
    )
    np.testing.assert_allclose(
        bias_gradient, finite_difference(bias, conv_loss), atol=1.0e-7
    )

    pool_input = np.array(
        [1.0, 2.0, 2.0, 3.0, 3.0, 0.0, 3.0, 1.0, 4.0]
    ).reshape(1, 1, 3, 3)
    pool_output_gradient = np.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    np.testing.assert_array_equal(
        max_pool2d_backward(
            pool_input, pool_output_gradient, (2, 2), (1, 0, 1, 0), (1, 1)
        ),
        np.array([1.0, 5.0, 0.0, 24.0, 6.0, 0.0, 0.0, 0.0, 9.0]).reshape(
            1, 1, 3, 3
        ),
    )

    smooth_pool_input = np.array(
        [1.0, 2.0, 3.0, 4.0, 8.0, 6.0, 7.0, 5.0, 9.0]
    ).reshape(1, 1, 3, 3)
    smooth_pool_output_gradient = np.array([1.0, 2.0, 3.0, 4.0]).reshape(
        1, 1, 2, 2
    )
    pool_loss = lambda: np.sum(
        max_pool2d(smooth_pool_input, (2, 2), (0, 0, 0, 0), (1, 1))
        * smooth_pool_output_gradient
    )
    np.testing.assert_allclose(
        max_pool2d_backward(
            smooth_pool_input,
            smooth_pool_output_gradient,
            (2, 2),
            (0, 0, 0, 0),
            (1, 1),
        ),
        finite_difference(smooth_pool_input, pool_loss),
        atol=1.0e-8,
    )

    print("NumPy CNN backward expected gradients and finite differences: ok")


if __name__ == "__main__":
    main()
