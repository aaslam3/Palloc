#pragma once

#include <cstddef>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#include <linux/mman.h>
#endif

inline constexpr bool palloc_is_windows =
#ifdef _WIN32
    true;
#else
    false;
#endif

// Portable compiler hints for cold/noinline functions.
// Keeps rarely-executed code out of the hot path's icache footprint.
#if defined(__GNUC__) || defined(__clang__)
#define PALLOC_COLD __attribute__((noinline, cold))
#elif defined(_MSC_VER)
#define PALLOC_COLD __declspec(noinline)
#else
#define PALLOC_COLD
#endif

namespace AL
{

constexpr size_t ONE_KB = 1024;
constexpr size_t ONE_MB = 1024 * ONE_KB;
constexpr size_t ONE_GB = 1024 * ONE_MB;
constexpr size_t ONE_TB = 1024 * ONE_GB;

//
// replaces platform specific system calls with a wrapper that changes which function is called based on what system you compiled for.
// has zero runtime overhead
//
struct platform_mem
{
    [[nodiscard]] static void* alloc(std::size_t size) noexcept
    {
#ifdef _WIN32
        return VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        return ptr == MAP_FAILED ? nullptr : ptr;
#endif
    }

    // allocates virtual memory without using physical memory
    // size is in bytes and must be a multiple of the system page size
    [[nodiscard]] static void* virtual_alloc(std::size_t size) noexcept
    {
#ifdef _WIN32
        return alloc(size);
#else
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | /*MAP_HUGETLB | MAP_HUGE_2MB |*/ MAP_NORESERVE, -1, 0);
        return ptr == MAP_FAILED ? nullptr : ptr;
#endif
    }

    static bool free(void* ptr, std::size_t size) noexcept
    {
#ifdef _WIN32
        (void)size;
        return VirtualFree(ptr, 0, MEM_RELEASE) != 0;
#else
        return munmap(ptr, size) == 0;
#endif
    }

    static std::size_t page_size() noexcept
    {
#ifdef _WIN32
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        return static_cast<std::size_t>(info.dwPageSize);
#else
        return static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#endif
    }
};

} // namespace AL
