# Authors: marcusm117
# License: Apache 2.0


# Standard Library Modules
import gzip


# uncompress a .gz compressed file
def g_unzip(input_path: str, output_path: str):
    """Uncompress a .gz compressed file.

    Args:
        input_path: the path to the .gz compressed file
        output_path: the path to the uncompressed file

    """
    with open(input_path, "rb") as reader, open(output_path, "w", encoding="utf8") as writer:
        decompressed_str = gzip.decompress(reader.read()).decode("utf-8")
        writer.write(decompressed_str)
