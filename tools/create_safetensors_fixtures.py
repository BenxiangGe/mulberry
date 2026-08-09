#!/usr/bin/env python3

import json
import struct
from pathlib import Path


PREFIX = Path("/tmp/mulberry_safetensors_validation_jit")


def write_tensor_file(path, header, data=b""):
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def tensor_header(shape, offsets):
    value = {
        "dtype": "F32",
        "shape": shape,
        "data_offsets": offsets,
    }
    return json.dumps({"x": value}, separators=(",", ":")).encode()


def main():
    PREFIX.with_name(PREFIX.name + "_short.safetensors").write_bytes(b"\x00")
    PREFIX.with_name(PREFIX.name + "_huge_header.safetensors").write_bytes(
        struct.pack("<Q", 100_000_001)
    )
    PREFIX.with_name(PREFIX.name + "_truncated_header.safetensors").write_bytes(
        struct.pack("<Q", 4)
    )
    write_tensor_file(
        PREFIX.with_name(PREFIX.name + "_overflow_shape.safetensors"),
        tensor_header([18_446_744_073_709_551_615, 2], [0, 0]),
    )
    write_tensor_file(
        PREFIX.with_name(PREFIX.name + "_numel_mismatch.safetensors"),
        tensor_header([2], [0, 4]),
    )
    write_tensor_file(
        PREFIX.with_name(PREFIX.name + "_data_out_of_bounds.safetensors"),
        tensor_header([2], [0, 8]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
