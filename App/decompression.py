import sys
import struct
import numpy as np
from PIL import Image

def unpack_indices(
    packed,
    number_of_indices,
    bits_per_index
):

    total_bits = number_of_indices * bits_per_index

    binary_data = np.unpackbits(
        packed,
        bitorder="big"
    )[:total_bits]

    binary_data = binary_data.reshape(
        number_of_indices,
        bits_per_index
    )

    shifts = np.arange(
        bits_per_index - 1,
        -1,
        -1
    )

    indices = np.sum(
        binary_data.astype(np.int64) * (1 << shifts),
        binary_data.astype(np.int64) * (1 << shifts),
        axis=1
    )

    return indices.astype(np.int32)


def read_compressed_file(filename):
def read_compressed_file(filename):

    with open(filename, "rb") as file:

        header_size = struct.calcsize(
            ">4sBIIIIIIIQ"
        )

        header = file.read(header_size)

        (
            magic,
            version,
            original_height,
            original_width,
            padded_height,
            padded_width,
            block_height,
            block_width,
            codebook_size,
            number_of_indices
        ) = struct.unpack(
            ">4sBIIIIIIIQ",
            header
        )

        if magic != b"VQ01":
            raise ValueError("Invalid compressed file.")

        if version != 1:
            raise ValueError("Unsupported file version.")

        bits_per_index = struct.unpack(
            ">B",
            file.read(1)
        )[0]

        vector_dimension = block_height * block_width

        codebook_bytes = file.read(
            codebook_size * vector_dimension * 4
        )

        codebook = np.frombuffer(
            codebook_bytes,
            dtype=np.float32
        ).reshape(
            codebook_size,
            vector_dimension
        )

        packed_size = (
            number_of_indices * bits_per_index + 7
        ) // 8

        packed_indices = np.frombuffer(
            file.read(packed_size),
            dtype=np.uint8
        )

    indices = unpack_indices(
        packed_indices,
        number_of_indices,
        bits_per_index
    )

    metadata = {
        "original_shape": (original_height, original_width),
        "padded_shape": (padded_height, padded_width),
        "block_height": block_height,
        "block_width": block_width
    }

    return codebook, indices, metadata


def reconstruct_image(codebook, indices, metadata):

    original_height, original_width = metadata["original_shape"]
    padded_height, padded_width = metadata["padded_shape"]
    block_height = metadata["block_height"]
    block_width = metadata["block_width"]

    blocks = codebook[indices]

    blocks = blocks.reshape(
        padded_height // block_height,
        padded_width // block_width,
        block_height,
        block_width
    )

    padded_image = blocks.swapaxes(1, 2).reshape(
        padded_height,
        padded_width
    )

    return padded_image[:original_height, :original_width]


def load_compressed_file(filename):

    codebook, indices, metadata = read_compressed_file(filename)

    reconstructed = reconstruct_image(
        codebook,
        indices,
        metadata
    )

    caller_globals = sys._getframe(1).f_globals
    caller_globals["reconstructed"] = reconstructed

    return codebook, indices, metadata


def decompress(
    compressed_filename,
    output_filename="reconstructed.png"
):
    codebook, indices, metadata = read_compressed_file(
        compressed_filename
    )

    reconstructed = reconstruct_image(
        codebook,
        indices,
        metadata
    )

    reconstructed = np.clip(
        np.rint(reconstructed),
        0,
        255
    ).astype(np.uint8)

    image = Image.fromarray(reconstructed, mode="L")

    image.save(output_filename)

    return output_filename
