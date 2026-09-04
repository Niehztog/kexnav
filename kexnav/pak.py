"""Minimal reader for the Quake II PACK archive format.

Only what is needed to pull nav files out of a retail pak0.pak without
unpacking 1.7 GB of assets: the directory is at the tail, every entry is a
fixed 64-byte record, and file bodies are read on demand.
"""

import struct

_HEADER = struct.Struct("<4sii")
_ENTRY = struct.Struct("<56sii")

MAGIC = b"PACK"


class PakError(Exception):
    pass


class Pak:
    """An open .pak archive. Use as a context manager."""

    def __init__(self, path):
        self.path = str(path)
        self._fp = open(self.path, "rb")
        try:
            magic, dir_ofs, dir_len = _HEADER.unpack(self._fp.read(_HEADER.size))
            if magic != MAGIC:
                raise PakError(f"{self.path}: not a PACK file (magic {magic!r})")
            if dir_len % _ENTRY.size:
                raise PakError(f"{self.path}: directory length {dir_len} is not a multiple of {_ENTRY.size}")
            self._fp.seek(dir_ofs)
            blob = self._fp.read(dir_len)
        except Exception:
            self._fp.close()
            raise

        self.entries = {}
        for i in range(dir_len // _ENTRY.size):
            raw, ofs, length = _ENTRY.unpack_from(blob, i * _ENTRY.size)
            name = raw.split(b"\0")[0].decode("latin-1")
            self.entries[name] = (ofs, length)

    @property
    def names(self):
        return self.entries.keys()

    def read(self, name):
        ofs, length = self.entries[name]
        self._fp.seek(ofs)
        data = self._fp.read(length)
        if len(data) != length:
            raise PakError(f"{self.path}: truncated reading {name!r}")
        return data

    def close(self):
        self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
