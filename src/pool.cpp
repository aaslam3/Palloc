#include "pool.h"
#include "platform.h"
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <mutex>

namespace AL
{
pool::pool()
{
    clear();
}

pool::pool(size_t block_size, size_t block_count) : pool()
{
    init(block_size, block_count);
}

pool::pool(pool&& other) noexcept
    : m_region(other.m_region), m_region_size(other.m_region_size), m_view(other.m_view), m_free_count(other.m_free_count.load())
{
    other.clear();
}

pool& pool::operator=(pool&& other) noexcept
{
    if (this == &other)
        return *this;

    if (m_region != nullptr)
        AL::platform_mem::free(m_region, m_region_size);

    m_region = other.m_region;
    m_region_size = other.m_region_size;
    m_view = other.m_view;
    m_free_count.store(other.m_free_count.load());

    other.clear();
    return *this;
}

void pool::init(size_t block_size, size_t block_count)
{
    assert(m_region == nullptr && "pool likely already initialized correctly.");

    if (block_size < sizeof(void*))
    {
#if PALLOC_DEBUG
        std::cerr << "WARNING: Pool block size " << block_size << " is too small. "
                  << "Rounded up to " << sizeof(void*) << " bytes.\n";
#endif
        block_size = sizeof(void*);
    }

    block_size = std::bit_ceil(block_size);

    size_t page_size = AL::platform_mem::page_size();
    size_t region_needed = pool_view::required_region_size(block_size, block_count);
    m_region_size = ((region_needed + page_size - 1) / page_size) * page_size;

    void* ptr = AL::platform_mem::virtual_alloc(m_region_size);
    if (ptr == nullptr)
        throw std::bad_alloc();

    m_region = static_cast<std::byte*>(ptr);
    m_view.init_from_region(m_region, block_size, block_count);
    m_free_count.store(block_count, std::memory_order_relaxed);
}

void pool::init_from_region(void* base, size_t block_size, size_t block_count)
{
    assert(!m_view.is_initialized() && "pool likely already initialized");
    assert(m_region == nullptr && "pool already owns memory");

    // non-owning: m_region stays nullptr so destructor won't munmap
    m_view.init_from_region(base, block_size, block_count);
    m_free_count.store(block_count, std::memory_order_relaxed);
}

pool::~pool()
{
    if (m_region == nullptr)
        return;

    bool freed = AL::platform_mem::free(m_region, m_region_size);

#if PALLOC_DEBUG
    if (!freed)
        std::cerr << "WARNING: munmap failed in pool destructor\n";
#endif

    m_region = nullptr;
}

void* pool::alloc()
{
    std::lock_guard<pool_mutex> lock(m_mutex);
    check_asserts();

    void* ptr = m_view.alloc();
    if (ptr != nullptr)
        m_free_count.store(m_view.free_count(), std::memory_order_relaxed);
    return ptr;
}

size_t pool::alloc_batched_internal(size_t num_objects, void* out[])
{
    std::lock_guard<pool_mutex> lock(m_mutex);
    if (!out)
        return 0;

    check_asserts();

    size_t i = 0;
    for (; i < num_objects; ++i)
    {
        void* ptr = m_view.alloc();
        if (ptr == nullptr)
            break;
        out[i] = ptr;
    }
    m_free_count.store(m_view.free_count(), std::memory_order_relaxed);
    return i;
}

void* pool::calloc()
{
    void* ptr = alloc();
    if (ptr != nullptr)
        std::memset(ptr, 0, m_view.block_size());
    return ptr;
}

void pool::reset()
{
    std::lock_guard<pool_mutex> lock(m_mutex);
    check_asserts();
    m_view.reset();
    m_free_count.store(m_view.block_count(), std::memory_order_relaxed);
}

void pool::clear()
{
    m_region = nullptr;
    m_region_size = 0;
    m_view = pool_view{};
    m_free_count.store(0, std::memory_order_relaxed);
}

bool pool::owns(void* ptr) const
{
    return m_view.owns(ptr);
}

void pool::free(void* ptr)
{
    std::lock_guard<pool_mutex> lock(m_mutex);
    if (ptr == nullptr)
        return;

    check_asserts();
    assert(owns(ptr) && "Pointer does not belong to this pool");

    m_view.free(ptr);
    m_free_count.store(m_view.free_count(), std::memory_order_relaxed);
}

void pool::free_batched_internal(size_t num_objects, void* in[])
{
    std::lock_guard<pool_mutex> lock(m_mutex);
    if (!in)
        return;

    check_asserts();

    for (size_t i = 0; i < num_objects; ++i)
    {
        if (!in[i])
            continue;

        assert(owns(in[i]) && "Pointer does not belong to this pool");
        m_view.free(in[i]);
    }
    
    m_free_count.store(m_view.free_count(), std::memory_order_relaxed);
}

size_t pool::get_free_space() const
{
    return m_free_count.load(std::memory_order_relaxed) * m_view.block_size();
}

size_t pool::get_capacity() const
{
    return m_view.capacity();
}

size_t pool::get_block_size() const
{
    return m_view.block_size();
}

size_t pool::get_block_count() const
{
    return m_view.block_count();
}

void pool::check_asserts() const
{
#if PALLOC_DEBUG
    assert(m_view.is_initialized() && "pool not initialized correctly.");
#endif
}

} // namespace AL
