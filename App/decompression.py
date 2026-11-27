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
        binary_data * (1 << shifts),
        axis=1
    )

    return indices.astype(np.int32)


def load_compressed_file(filename):

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
