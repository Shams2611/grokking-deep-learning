"""
Chapter 10: Convolution from Scratch

Convolutions are the heart of CNNs (Convolutional Neural Networks)!

What's a convolution?
Slide a small "filter" (kernel) across an image, computing dot products.
The filter learns to detect features like edges, textures, shapes.

Why convolutions for images?
1. Translation invariance: detect a cat anywhere in the image
2. Parameter sharing: same filter used everywhere (fewer weights!)
3. Local connectivity: pixels far apart don't directly interact

Key terms:
- Kernel/Filter: small matrix of learnable weights
- Stride: how many pixels to skip when sliding
- Padding: add zeros around edges to control output size

This file implements convolution step by step so we understand it!
"""

import numpy as np


def convolve2d(image, kernel, stride=1, padding=0):
    """
    2D convolution operation.

    Slide kernel across image, computing dot products.

    Args:
        image: 2D input (height x width)
        kernel: 2D filter (kernel_h x kernel_w)
        stride: step size when sliding
        padding: zeros to add around edges

    Returns:
        output: convolved result
    """
    # Add padding if needed
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)

    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape

    # Calculate output size
    out_h = (img_h - ker_h) // stride + 1
    out_w = (img_w - ker_w) // stride + 1

    output = np.zeros((out_h, out_w))

    # Slide the kernel across the image
    for i in range(out_h):
        for j in range(out_w):
            # Extract the region under the kernel
            region = image[
                i * stride : i * stride + ker_h,
                j * stride : j * stride + ker_w
            ]

            # Dot product = element-wise multiply and sum
            output[i, j] = np.sum(region * kernel)

    return output


def convolve2d_multichannel(image, kernels, stride=1, padding=0):
    """
    Convolution with multiple input channels and multiple filters.

    image: (channels, height, width)
    kernels: (num_filters, channels, kernel_h, kernel_w)

    Returns: (num_filters, out_h, out_w)
    """
    num_filters = kernels.shape[0]
    channels = image.shape[0]

    # Get output size from first channel
    if padding > 0:
        padded = np.pad(image[0], padding, mode='constant')
    else:
        padded = image[0]

    out_h = (padded.shape[0] - kernels.shape[2]) // stride + 1
    out_w = (padded.shape[1] - kernels.shape[3]) // stride + 1

    output = np.zeros((num_filters, out_h, out_w))

    # For each filter
    for f in range(num_filters):
        # Sum contributions from each channel
        for c in range(channels):
            output[f] += convolve2d(image[c], kernels[f, c], stride, padding)

    return output


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("2D Convolution from Scratch")
    print("=" * 60)

    # Simple example
    print("\n--- Simple 4x4 Image ---")

    image = np.array([
        [1, 2, 3, 0],
        [0, 1, 2, 3],
        [3, 0, 1, 2],
        [2, 3, 0, 1]
    ], dtype=float)

    print("Input image:")
    print(image)

    # Edge detection kernel
    edge_kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=float)

    print("\nEdge detection kernel:")
    print(edge_kernel)

    result = convolve2d(image, edge_kernel, stride=1, padding=0)
    print("\nConvolution result:")
    print(result)

    # Step-by-step trace
    print("\n" + "=" * 60)
    print("Step-by-step trace (first output position):")
    print("=" * 60)

    print("\nRegion under kernel (top-left 3x3 of image):")
    region = image[0:3, 0:3]
    print(region)

    print("\nKernel:")
    print(edge_kernel)

    print("\nElement-wise multiplication:")
    mult = region * edge_kernel
    print(mult)

    print(f"\nSum = {mult.sum()}")
    print(f"This is output[0,0] = {result[0,0]}")

    # Common kernels
    print("\n" + "=" * 60)
    print("Common Convolution Kernels")
    print("=" * 60)

    # Sample image with clear pattern
    test_img = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)

    print("\nTest image (diamond shape):")
    for row in test_img:
        print("".join(['#' if x > 0 else '.' for x in row]))

    kernels = {
        "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        "Edge detect": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "Blur": np.ones((3, 3)) / 9,
        "Vertical edge": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        "Horizontal edge": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    }

    for name, kernel in kernels.items():
        result = convolve2d(test_img, kernel, stride=1, padding=1)
        print(f"\n{name}:")
        # Show result as ASCII
        max_val = max(abs(result.min()), abs(result.max()), 0.01)
        for row in result:
            line = ""
            for val in row:
                if val > max_val * 0.3:
                    line += "#"
                elif val < -max_val * 0.3:
                    line += "-"
                else:
                    line += "."
            print(line)

    print("\n" + "=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. Convolution = slide kernel, compute dot products")
    print("2. Kernels detect features (edges, textures, etc.)")
    print("3. In CNNs, kernels are LEARNED, not hand-designed")
    print("4. Padding controls output size")
    print("5. Stride controls how much we skip")
    print("6. Multiple filters -> multiple feature maps")
