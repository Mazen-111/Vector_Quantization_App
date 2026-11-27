import compression as comp
import decompression as decomp
from PIL import Image
import numpy as np

def compress(image_path, block_height, block_width, codebook_size):
    return comp.compress(image_path, block_height, block_width, codebook_size)

def decompress(compressed_file):
    return decomp.decompress(compressed_file)

def main():
    print("===================================================")
    print("======== Welcome to Vector Quantization App =======")
    print("===================================================")
    choice = input("Enter 'c' to compress or 'd' to decompress: ").strip().lower()
    if choice == 'c':
        path = input("Enter the path of the image to compress: ").strip()
        v_h = int(input("Enter the block height: ").strip())
        v_w = int(input("Enter the block width: ").strip())
        c_s = int(input("Enter the codebook size: ").strip())
        compressed_file = compress(path, v_h, v_w, c_s)
        print(f"Image compressed and saved to {compressed_file}")
        choice = input("Do you want to decompress the image? (y/n): ").strip().lower()
        if choice == 'y':
            decompress(compressed_file)
        else:
            print("Thank you for using the Vector Quantization App.")
    
    elif choice == 'd':
        path = input("Enter the path of the compressed file to decompress: ").strip()
        output_file = decompress(path)
        print(f"Image decompressed and saved to {output_file}")
        print("Thank you for using the Vector Quantization App.")

main()