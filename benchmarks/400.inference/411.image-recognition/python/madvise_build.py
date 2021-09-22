from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("""
    int usm_madvise(unsigned long addr, size_t length);
""")

ffibuilder.set_source("_mman_cffi",
"""
    #include <sys/mman.h>
    #include <stdint.h>

    static int usm_madvise(unsigned long addr, size_t length) {
        return madvise((void *)addr, length, 20);
    }
""",
    libraries=[])

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
