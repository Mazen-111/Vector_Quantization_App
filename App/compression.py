from PIL import Image
import numpy as np
import struct
import os

def load_image(path):
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float64)


def divide_into_vectors(image, block_height, block_width):
    height, width = image.shape

    padded_height = (
        height + (block_height - height % block_height) % block_height
    )

    padded_width = (
        width + (block_width - width % block_width) % block_width
    )

    padded_image = np.pad(
        image,
        (
            (0, padded_height - height),
            (0, padded_width - width)
        ),
        mode="edge"
    )

    vectors = []

    for i in range(0, padded_height, block_height):
        for j in range(0, padded_width, block_width):

            block = padded_image[
                i:i + block_height,
                j:j + block_width
            ]

            vectors.append(block.flatten())

    return np.array(vectors), padded_image.shape


def calculate_distances(vectors, codebook):

    distances = np.sum(
        np.abs(
            vectors[:, np.newaxis, :] -
            codebook[np.newaxis, :, :]
        ),
        axis=2
    )

    return distances


def assign_vectors(vectors, codebook, batch_size=50000):

    assignments = np.empty(len(vectors), dtype=np.int32)

    for start in range(0, len(vectors), batch_size):

        end = min(start + batch_size, len(vectors))

        batch = vectors[start:end]

        distances = calculate_distances(batch, codebook)

        assignments[start:end] = np.argmin(
            distances,
            axis=1
        )

    return assignments


def update_centroids(vectors, assignments, codebook):

    k = len(codebook)
    dimensions = vectors.shape[1]

    sums = np.zeros((k, dimensions), dtype=np.float64)

    counts = np.bincount(
        assignments,
        minlength=k
    )

    np.add.at(sums, assignments, vectors)

    new_codebook = codebook.copy()

    for i in range(k):

        if counts[i] > 0:

            new_codebook[i] = sums[i] / counts[i]

    return new_codebook


def lbg(
    vectors,
    codebook_size,
    delta=1.0,
    tolerance=0.01,
    max_iterations=100
):
    codebook = np.mean(
        vectors,
        axis=0,
        keepdims=True
    )

    while len(codebook) < codebook_size:

        codebook = np.vstack([
            codebook - delta,
            codebook + delta
        ])

        for _ in range(max_iterations):

            assignments = assign_vectors(
                vectors,
                codebook
            )

            new_codebook = update_centroids(
                vectors,
                assignments,
                codebook
            )

            change = np.mean(
                np.abs(new_codebook - codebook)
            )

            codebook = new_codebook

            if change < tolerance:
                break

    return codebook


def encode(vectors, codebook):

    indices = assign_vectors(
        vectors,
        codebook
    )

    return indices


def pack_indices(indices, codebook_size):

    bits_per_index = max(
        1,
        (codebook_size - 1).bit_length()
    )

    shifts = np.arange(
        bits_per_index - 1,
        -1,
        -1
    )

    binary_indices = (
        (indices[:, None] >> shifts) & 1
    ).astype(np.uint8)

    packed = np.packbits(
        binary_indices.flatten(),
        bitorder="big"
    )

    return packed, bits_per_index

def save_compressed_file(
    filename,
    codebook,
    indices,
    original_shape,
    padded_shape,
    block_height,
    block_width
):

    original_height, original_width = original_shape
    padded_height, padded_width = padded_shape

    codebook_size = len(codebook)
    number_of_indices = len(indices)

    packed_indices, bits_per_index = pack_indices(
        indices,
        codebook_size
    )

    codebook_data = codebook.astype(np.float32)

    with open(filename, "wb") as file:

        file.write(
            struct.pack(
                ">4sBIIIIIIIQ",
                b"VQ01",
                1,
                original_height,
                original_width,
                padded_height,
                padded_width,
                block_height,
                block_width,
                codebook_size,
                number_of_indices
            )
        )

        file.write(
            struct.pack(">B", bits_per_index)
        )

        file.write(
            codebook_data.tobytes()
        )

        file.write(
            packed_indices.tobytes()
        )
        

def calculate_metrics(original_image, block_height, block_width, codebook_size, compressed_filename):
    original_size = original_image.size
    num_blocks = (original_image.shape[0] // block_height) * (original_image.shape[1] // block_width)
    labels_size = num_blocks * (codebook_size - 1).bit_length()
    total_compressed_size = os.path.getsize(compressed_filename)
    compression_ratio = total_compressed_size / original_size
    print("================== Compression Metrics ==================")
    print(f"Original size = {original_size} bytes")
    print(f"Number of blocks = {num_blocks}")
    print(f"Labels size = {labels_size} bits")
    print(f"Compressed size = {total_compressed_size} bytes")
    print(f"Compression ratio = {compression_ratio:.2f}")
    print("=========================================================")

def compress(
    image_path,
    block_height,
    block_width,
    codebook_size,
    compressed_filename="compressed.vq",
    delta=1.0,
    tolerance=0.01,
    max_iterations=100
):
    image = load_image(image_path)

    original_shape = image.shape

    vectors, padded_shape = divide_into_vectors(
        image,
        block_height,
        block_width
    )

    codebook = lbg(
        vectors,
        codebook_size,
        delta,
        tolerance,
        max_iterations
    )

    indices = encode(vectors, codebook)

    save_compressed_file(
        compressed_filename,
        codebook,
        indices,
        original_shape,
        padded_shape,
        block_height,
        block_width
    )
    
    calculate_metrics(image, block_height, block_width, codebook_size, compressed_filename)
    
    return compressed_filename