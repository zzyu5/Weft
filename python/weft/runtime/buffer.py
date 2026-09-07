from __future__ import annotations

import ctypes
import weakref
from collections.abc import Sequence


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p), ("obj", ctypes.c_void_p),
        ("len", ctypes.c_ssize_t), ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int), ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p), ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)), ("internal", ctypes.c_void_p),
    ]


_get_buffer = ctypes.pythonapi.PyObject_GetBuffer
_get_buffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_get_buffer.restype = ctypes.c_int
_release_buffer = ctypes.pythonapi.PyBuffer_Release
_release_buffer.argtypes = [ctypes.POINTER(_PyBuffer)]
_release_buffer.restype = None

_DENSE_FORMATS = {
    "b": "i8", "B": "u8", "h": "i16", "H": "u16",
    "i": "i32", "I": "u32", "q": "i64", "Q": "u64",
    "e": "f16", "f": "f32", "d": "f64", "?": "i1",
}


class Buffer:
    """A borrowed contiguous storage object with an explicit logical Encoding."""

    def __init__(self, data: object, *, shape: Sequence[int] | None = None,
                 encoding: str | None = None) -> None:
        view = memoryview(data)
        if not view.c_contiguous:
            raise ValueError("the current kernel ABI requires contiguous Encoding storage")
        if encoding is None:
            spelling = view.format.lstrip("@=<")
            if spelling in ("l", "L") and view.itemsize in (4, 8):
                encoding = "dense." + ("i" if spelling == "l" else "u") + str(view.itemsize * 8)
            elif spelling in _DENSE_FORMATS:
                encoding = "dense." + _DENSE_FORMATS[spelling]
            else:
                raise TypeError(f"buffer format {view.format!r} needs an explicit Encoding")
        if not isinstance(encoding, str) or not encoding:
            raise TypeError("buffer Encoding must be a non-empty layout identity")
        logical_shape = tuple(view.shape if shape is None else shape)
        if any(type(extent) is not int or extent < 0 for extent in logical_shape):
            raise ValueError("logical buffer extents must be non-negative integers")
        storage = _PyBuffer()
        _get_buffer(view, ctypes.byref(storage), 0)
        self._release = weakref.finalize(self, _release_buffer, ctypes.byref(storage))
        self._storage = storage
        self._view = view
        self.shape = logical_shape
        self.encoding = encoding
        self.address = storage.buf
        self.nbytes = storage.len
        self.readonly = bool(storage.readonly)
