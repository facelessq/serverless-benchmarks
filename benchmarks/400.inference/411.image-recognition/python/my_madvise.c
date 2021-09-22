#include <sys/mman.h>
#include <stdint.h>

#ifdef _MSC_VER
    #define DLL_EXPORT __declspec( dllexport ) 
#else
    #define DLL_EXPORT
#endif

DLL_EXPORT int usm_madvise(int addr, size_t length) {
    return madvise((void *)(intptr_t)addr, length, 20);
}
