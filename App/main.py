import compression
import decompression
from PIL import Image
import numpy as np

image_path = "luffy.jpg"

block_height = 2
block_width = 2

codebook_size = 16

image = compression.load_image(image_path)

original_shape = image.shape

print("Original image shape:", original_shape)

vectors, padded_shape = compression.divide_into_vectors(
    image,
    block_height,
    block_width
)

print("Number of vectors:", len(vectors))
print("Vector dimension:", vectors.shape[1])

codebook = compression.lbg(
    vectors,
    codebook_size
)

print("Codebook shape:", codebook.shape)

indices = compression.encode(
    vectors,
    codebook
)

print("Number of encoded indices:", len(indices))

compression.save_compressed_file(
    "compressed_image.vqbin",
    codebook,
    indices,
    original_shape,
    padded_shape,
    block_height,
    block_width
)

print("Compressed binary file saved.")

loaded_codebook, loaded_indices, metadata = (
    compression.load_compressed_file("compressed_image.vqbin")
)

# code to return reconstructed array

result = Image.fromarray(
    np.clip(
        reconstructed,
        0,
        255
    ).astype(np.uint8)
)

result.save("reconstructed.png")

result.show()

# Calculate MSE
mse = np.mean(
    (image - reconstructed) ** 2
)

print("MSE:", mse)

print("Compression and reconstruction completed!")
