import ctypes
from numpy import ndarray
import numpy as np


class Vector:
    def __init__(self, diffeq, array: list | ndarray):
        self.pointer = diffeq.Vector_create_with_capacity(0, len(array))
        push = diffeq.Vector_push
        for i in range(len(array)):
            push(self.pointer, np.float64(array[i]))
        self.diffeq = diffeq

    def get(self, index):
        return self.diffeq.Vector_get(self.pointer, index)

    def getFloat64Array(self) -> ndarray:
        length = self.diffeq.Vector_get_length(self.pointer)
        data = self.diffeq.Vector_get_data(self.pointer)
        ptr_type = ctypes.c_double * length
        mem_address = ctypes.addressof(self.diffeq.memory_ptr().contents) + data
        buffer = ptr_type.from_address(mem_address)
        return np.frombuffer(buffer, dtype=np.float64)

    def destroy(self):
        self.diffeq.Vector_destroy(self.pointer)

    def resize(self, len):
        self.diffeq.Vector_resize(self.pointer, len)

    def length(self):
        return self.diffeq.Vector_get_length(self.pointer)

    def __len__(self):
        return self.length()
