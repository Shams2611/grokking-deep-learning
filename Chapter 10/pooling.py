"""
Chapter 10: Pooling Layers

Pooling reduces the spatial size of feature maps!

Why pooling?
1. Reduces computation (fewer pixels to process)
2. Provides translation invariance (small shifts don't matter)
3. Helps prevent overfitting (fewer parameters)

Types:
- Max pooling: take the maximum value in each region
- Average pooling: take the average value in each region

Max pooling is more common because it preserves important features
(the biggest activations are usually the most important).

Typical usage: 2x2 max pooling with stride 2
- Cuts height and width in HALF
- Keeps the strongest feature in each 2x2 region
"""

import numpy as np


def max_pool2d(image, pool_size=2, stride=2):
    """
    Max pooling operation.

    Divide image into regions, take max of each region.

    Args:
        image: 2D input (height x width)
        pool_size: size of pooling window
        stride: step size (usually equals pool_size)

    Returns:
        pooled output (smaller than input!)
    """
    img_h, img_w = image.shape

    out_h = (img_h - pool_size) // stride + 1
    out_w = (img_w - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # Extract region
            region = image[
                i * stride : i * stride + pool_size,
                j * stride : j * stride + pool_size
            ]

            # Take maximum
            output[i, j] = np.max(region)

    return output


def avg_pool2d(image, pool_size=2, stride=2):
    """
    Average pooling operation.

    Same as max pool but takes average instead of max.
    """
    img_h, img_w = image.shape

    out_h = (img_h - pool_size) // stride + 1
    out_w = (img_w - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = image[
                i * stride : i * stride + pool_size,
                j * stride : j * stride + pool_size
            ]

            output[i, j] = np.mean(region)

    return output


def max_pool2d_with_indices(image, pool_size=2, stride=2):
    """
    Max pooling that also returns WHERE the max came from.

    We need this for backpropagation!
    The gradient only flows to the position that had the max.
    """
    img_h, img_w = image.shape

    out_h = (img_h - pool_size) // stride + 1
    out_w = (img_w - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))
    indices = np.zeros((out_h, out_w, 2), dtype=int)

    for i in range(out_h):
        for j in range(out_w):
            i_start = i * stride
            j_start = j * stride

            region = image[
                i_start : i_start + pool_size,
                j_start : j_start + pool_size
            ]

            # Find max and its position
            max_val = np.max(region)
            max_pos = np.unravel_index(np.argmax(region), region.shape)

            output[i, j] = max_val
            # Store absolute position
            indices[i, j] = [i_start + max_pos[0], j_start + max_pos[1]]

    return output, indices


# ============================================
# Demo
# ============================================
if __name__ == "__main__":

    print("=" * 60)
    print("Pooling Layers")
    print("=" * 60)

    # Simple example
    print("\n--- 4x4 Input ---")

    image = np.array([
        [1, 2, 5, 4],
        [3, 8, 2, 1],
        [4, 1, 6, 3],
        [2, 5, 3, 9]
    ], dtype=float)

    print("Input:")
    print(image)

    # Max pooling
    max_pooled = max_pool2d(image, pool_size=2, stride=2)
    print("\nMax pooling (2x2, stride 2):")
    print(max_pooled)

    # Average pooling
    avg_pooled = avg_pool2d(image, pool_size=2, stride=2)
    print("\nAverage pooling (2x2, stride 2):")
    print(avg_pooled)

    # Step-by-step trace
    print("\n" + "=" * 60)
    print("Step-by-step Max Pooling:")
    print("=" * 60)

    print("\nInput divided into 2x2 regions:")
    print("  +-------+-------+")
    print(f"  | {image[0,0]} {image[0,1]} | {image[0,2]} {image[1,3]} |")
    print(f"  | {image[1,0]} {image[1,1]} | {image[1,2]} {image[1,3]} |")
    print("  +-------+-------+")
    print(f"  | {image[2,0]} {image[2,1]} | {image[2,2]} {image[2,3]} |")
    print(f"  | {image[3,0]} {image[3,1]} | {image[3,2]} {image[3,3]} |")
    print("  +-------+-------+")

    print("\nMax of each region:")
    print(f"  Top-left:     max({image[0,0]}, {image[0,1]}, {image[1,0]}, {image[1,1]}) = {max_pooled[0,0]}")
    print(f"  Top-right:    max({image[0,2]}, {image[0,3]}, {image[1,2]}, {image[1,3]}) = {max_pooled[0,1]}")
    print(f"  Bottom-left:  max({image[2,0]}, {image[2,1]}, {image[3,0]}, {image[3,1]}) = {max_pooled[1,0]}")
    print(f"  Bottom-right: max({image[2,2]}, {image[2,3]}, {image[3,2]}, {image[3,3]}) = {max_pooled[1,1]}")

    # Translation invariance demo
    print("\n" + "=" * 60)
    print("Translation Invariance Demo")
    print("=" * 60)

    # Create a small "feature" at different positions
    img1 = np.zeros((4, 4))
    img1[0, 0] = 5  # feature in top-left

    img2 = np.zeros((4, 4))
    img2[1, 1] = 5  # feature shifted slightly

    print("\nImage 1 (feature at 0,0):")
    print(img1)
    print(f"Max pool result: {max_pool2d(img1)}")

    print("\nImage 2 (feature at 1,1 - shifted!):")
    print(img2)
    print(f"Max pool result: {max_pool2d(img2)}")

    print("\nSame output! Pooling makes small shifts invisible.")
    print("This helps CNNs recognize objects regardless of exact position.")

    # Size reduction demo
    print("\n" + "=" * 60)
    print("Size Reduction")
    print("=" * 60)

    sizes = [(28, 28), (14, 14), (7, 7)]

    print("\nTypical CNN progression (like in image classification):")
    print("  28x28 -> Conv -> 28x28 -> Pool -> 14x14")
    print("  14x14 -> Conv -> 14x14 -> Pool -> 7x7")
    print("  7x7   -> Conv -> 7x7   -> Pool -> 3x3")
    print("\nEach pool layer cuts dimensions in half!")

    big_img = np.random.randn(28, 28)
    pooled_once = max_pool2d(big_img)
    pooled_twice = max_pool2d(pooled_once)

    print(f"\nOriginal: {big_img.shape}")
    print(f"After 1 pool: {pooled_once.shape}")
    print(f"After 2 pools: {pooled_twice.shape}")

    print("\n" + "=" * 60)
    print("Key Points:")
    print("=" * 60)
    print("1. Pooling reduces spatial dimensions")
    print("2. Max pooling: keep strongest activation in each region")
    print("3. Average pooling: keep average activation")
    print("4. Provides translation invariance")
    print("5. Typical: 2x2 pooling with stride 2 (halves dimensions)")
    print("6. For backprop: gradient only goes to max position")
