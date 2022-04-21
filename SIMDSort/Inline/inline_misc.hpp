#include <intrin.h>

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

// bitscanforward
__forceinline unsigned int bsf(unsigned int n) {
#ifdef _DEBUG
    if (n == 0u) {
        throw std::exception("undefined bsf");
    }
#endif // _DEBUG

#ifdef _MSC_VER
    unsigned long index;

    _BitScanForward(&index, n);

    return (unsigned int)index;
#else
    return __builtin_ctz(n);
#endif
}