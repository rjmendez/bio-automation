#!/usr/bin/env python3
"""
Topaz Photo AI .tz2 model file decryptor.

ALL PARAMETERS VERIFIED via Frida runtime instrumentation of aiengine.dll v2.9.4.

============================================================
ENCRYPTION SCHEME (confirmed by static + dynamic analysis)
============================================================

1. HKDF-SHA256 key derivation (OpenSSL 3.x EVP_KDF):
     IKM    = b"a0331add513e4e6f"  (16 ASCII bytes, hardcoded in .rdata of aiengine.dll)
     Salt   = file[18:34]          (16 bytes from file header, per-file random)
     Info   = b"payload"           (7 ASCII bytes, hardcoded in .rdata of aiengine.dll)
     Output = 32 bytes             (ChaCha20-Poly1305 key)

2. ChaCha20-Poly1305 block cipher (OpenSSL EVP_chacha20_poly1305):
     Block size = 67108864 bytes (64 MiB), each block independently authenticated
     Nonce for block N (0-indexed, 12 bytes total):
       bytes  0-9  = 0x00 (always zero)
       byte  10    = N & 0xFF (block index, lower byte)
       byte  11    = 0x01 if N is the LAST block, else 0x00

3. File layout:
     [0:18]   "com.topazlabs.tz2\x00"  magic (18 bytes)
     [18:34]  per-file HKDF salt        (16 bytes, randomly generated at encrypt time)
     [34:]    blocks: ciphertext_N || poly1305_tag_N  (each block at most 64 MiB + 16B tag)

============================================================
USAGE
============================================================
  pip install cryptography
  python decrypt_tz2.py <input.tz2> <output_file> [-v]

============================================================
EMPIRICAL VERIFICATION
============================================================
  Frida runtime hook of aiengine.dll v2.9.4 confirmed:
    - EVP_KDF_derive called with OSSL_PARAM: digest=SHA256, key=a0331add513e4e6f (16B),
      salt=<file[18:34]>, info=payload
    - EVP_CipherInit_ex called with derived key and 12-byte IV
    - Nonce pattern confirmed on 105+ model files spanning 1-25+ blocks
"""

import sys
import math
from pathlib import Path

try:
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
except ImportError:
    print("Install: pip install cryptography")
    sys.exit(1)

MAGIC      = b"com.topazlabs.tz2\x00"
MAGIC_LEN  = 18
SALT_LEN   = 16     # only the first 16 bytes of the 32-byte field are the HKDF salt
HDR_LEN    = MAGIC_LEN + SALT_LEN   # = 34 bytes
BLOCK_SIZE = 64 * 1024 * 1024       # 67,108,864 bytes per block
TAG_SIZE   = 16                      # Poly1305 authentication tag

IKM  = b"a0331add513e4e6f"   # 16 ASCII bytes (hardcoded IKM)
INFO = b"payload"             # 7 ASCII bytes  (hardcoded HKDF info)


def _derive_key(salt16: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt16,
        info=INFO,
    ).derive(IKM)


def _block_nonce(block_index: int, is_last: bool) -> bytes:
    """
    12-byte ChaCha20-Poly1305 nonce for block N.
      bytes 0-9:  always 0x00
      byte 10:    block_index & 0xFF
      byte 11:    0x01 if this is the last block, else 0x00
    """
    nonce = bytearray(12)
    nonce[10] = block_index & 0xFF
    nonce[11] = 0x01 if is_last else 0x00
    return bytes(nonce)


def decrypt_tz2(data: bytes, verbose: bool = False) -> bytes:
    """Decrypt a complete .tz2 blob."""
    if not data.startswith(MAGIC):
        raise ValueError(f"Not a .tz2 file (bad magic): {data[:20]!r}")

    salt = data[MAGIC_LEN : HDR_LEN]      # bytes 18-33
    ct_all = data[HDR_LEN:]               # ciphertext stream starts at byte 34

    key = _derive_key(salt)
    cipher = ChaCha20Poly1305(key)

    if verbose:
        print(f"  Salt (16B): {salt.hex()}")
        print(f"  Key  (32B): {key.hex()}")
        print(f"  CT length:  {len(ct_all):,} bytes")

    # Calculate number of blocks
    total_bytes = len(ct_all)
    full_blocks, remainder = divmod(total_bytes, BLOCK_SIZE + TAG_SIZE)
    # If there's a remainder (partial block with at least tag_size+1 bytes):
    n_blocks = full_blocks + (1 if remainder > TAG_SIZE else 0)
    if n_blocks == 0:
        raise ValueError("No ciphertext blocks found")

    plaintext_chunks = []
    offset = 0

    for block_idx in range(n_blocks):
        is_last = (block_idx == n_blocks - 1)
        block_end = min(offset + BLOCK_SIZE + TAG_SIZE, total_bytes)
        block_ct = ct_all[offset:block_end]

        if len(block_ct) <= TAG_SIZE:
            raise ValueError(f"Block {block_idx} too small: {len(block_ct)} bytes")

        nonce = _block_nonce(block_idx, is_last)

        try:
            pt = cipher.decrypt(nonce, block_ct, None)
        except Exception as e:
            raise ValueError(
                f"Block {block_idx} auth failure (nonce={nonce.hex()}): {e}"
            )

        plaintext_chunks.append(pt)

        if verbose:
            print(f"  Block {block_idx}: {len(block_ct):,}B CT -> {len(pt):,}B PT "
                  f"(nonce=...{nonce[-4:].hex()}, last={is_last})")

        offset = block_end

    return b"".join(plaintext_chunks)


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.tz2> <output_file> [-v]")
        sys.exit(1)

    inpath  = Path(sys.argv[1])
    outpath = Path(sys.argv[2])
    verbose = "-v" in sys.argv

    data = inpath.read_bytes()
    print(f"Decrypting {inpath.name} ({len(data):,} bytes)")

    try:
        plaintext = decrypt_tz2(data, verbose=verbose)
        outpath.write_bytes(plaintext)
        print(f"[OK] {outpath} ({len(plaintext):,} bytes)")
        print(f"     first 32B: {plaintext[:32].hex()}")
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
