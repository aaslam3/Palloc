#pragma once

#include "palloc_atomic.h"
#include "platform.h"
#include "pool.h"
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>

namespace AL
{

struct size_class
{
    std::size_t byte_size;
    std::size_t num_blocks;
    std::size_t batch_size;
};

constexpr bool is_power_of_two(std::size_t x) noexcept
{
    return x && ((x & (x - 1)) == 0);
}

consteval bool is_valid_config(const auto& arr) noexcept
{
    std::size_t prev_size = 0;
    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        auto const& sc = arr[i];
        if (sc.byte_size == 0)
            return false;
        if (!is_power_of_two(sc.byte_size))
            return false;
        if (sc.num_blocks == 0)
            return false;
        if (sc.batch_size == 0)
            return false;
        if (sc.batch_size > sc.num_blocks)
            return false;
        if (i > 0 && sc.byte_size <= prev_size)
            return false;
        if (sc.num_blocks > (std::numeric_limits<std::size_t>::max() / sc.byte_size))
            return false;
        prev_size = sc.byte_size;
    }
    return true;
}

template<std::size_t Tnum = 10,
         std::array<size_class, Tnum> Tsize_class_config =
             {
                 size_class{   .byte_size = 8, .num_blocks = 512, .batch_size = 64},
                 size_class{  .byte_size = 16, .num_blocks = 512, .batch_size = 64},
                 size_class{  .byte_size = 32, .num_blocks = 256, .batch_size = 32},
                 size_class{  .byte_size = 64, .num_blocks = 256, .batch_size = 32},
                 size_class{ .byte_size = 128, .num_blocks = 128, .batch_size = 16},
                 size_class{ .byte_size = 256, .num_blocks = 128, .batch_size = 16},
                 size_class{ .byte_size = 512,  .num_blocks = 64,  .batch_size = 8},
                 size_class{.byte_size = 1024,  .num_blocks = 64,  .batch_size = 8},
                 size_class{.byte_size = 2048,  .num_blocks = 32,  .batch_size = 4},
                 size_class{.byte_size = 4096,  .num_blocks = 32,  .batch_size = 4}
},
         std::size_t Tnum_cached_classes = Tnum>
struct slab_config
{
    static_assert(Tnum_cached_classes <= Tnum, "NUM_CACHED_CLASSES must be <= total size-class count");

    static_assert(Tnum > 0, "at least one size class required");
    static_assert(is_valid_config(Tsize_class_config),
                  "Invalid SIZE_CLASS_CONFIG: power-of-two, strictly increasing sizes, non-zero counts required");

    inline static constexpr std::array<size_class, Tnum> SIZE_CLASS_CONFIG = Tsize_class_config;
    static constexpr std::size_t NUM_SIZE_CLASSES = Tnum;
    static constexpr std::size_t NUM_CACHED_CLASSES = Tnum_cached_classes;

    static constexpr std::size_t INDEX_SPAN =
        std::bit_width(Tsize_class_config[Tnum - 1].byte_size) -
        std::bit_width(Tsize_class_config[0].byte_size) + 1;

    // gaps in the config round up to the next available size class.
    // e.g. in config {8,64}: sizes 9-64 all go to 64B pool.
    static consteval auto compute_index_lut()
    {
        std::array<std::size_t, INDEX_SPAN> lut{};
        std::size_t min_bw = std::bit_width(Tsize_class_config[0].byte_size);
        for (std::size_t vi = 0; vi < INDEX_SPAN; ++vi)
        {
            std::size_t target_size = std::size_t(1) << (min_bw + vi - 1);
            lut[vi] = static_cast<std::size_t>(-1);
            for (std::size_t j = 0; j < Tnum; ++j)
            {
                if (Tsize_class_config[j].byte_size >= target_size)
                {
                    lut[vi] = j;
                    break;
                }
            }
        }
        return lut;
    }

    inline static constexpr auto INDEX_LUT = compute_index_lut();

    // compute total bytes needed for all pools' sub-regions (with alignment padding).
    // assumes page-aligned base (mmap), so first pool always starts aligned.
    static constexpr std::size_t compute_total_region_size()
    {
        std::size_t total = 0;
        for (std::size_t i = 0; i < Tnum; ++i)
        {
            auto const& sc = Tsize_class_config[i];
            // align cursor to block_size
            std::size_t mask = sc.byte_size - 1;
            total = (total + mask) & ~mask;
            // add region for this pool
            std::size_t bitmap_words = (sc.num_blocks + 63) / 64;
            std::size_t bitmap_bytes = bitmap_words * sizeof(uint64_t);
            std::size_t aligned_offset = ((bitmap_bytes + sc.byte_size - 1) / sc.byte_size) * sc.byte_size;
            total += aligned_offset + sc.byte_size * sc.num_blocks;
        }
        return total;
    }
};

struct thread_local_cache
{
    // max capacity — actual batch sizes are tuned per size class
    static constexpr size_t object_count = 128;

    std::array<void*, object_count> objects;
    size_t current = 0;
    size_t batch_size = object_count / 2; // filled by slab on cache init

    [[nodiscard]] void* try_pop()
    {
        if (is_empty())
            return nullptr;

        current--;
        return objects[current];
    }

    void push(void* ptr)
    {
        assert(!is_full() && "Thread local cache is full");

        objects[current] = ptr;
        current++;
    }

    bool is_empty() const
    {
        return current == 0;
    }

    bool is_full() const
    {
        return current == object_count;
    }

    void invalidate()
    {
        current = 0;
    }
};

template<typename Tconfig>
class slab
{
    static constexpr size_t VIRTUAL_MEM_ALLOC_SIZE = AL::ONE_GB * 100; 
public:
    slab();
    ~slab();

    slab(const slab&) = delete;
    slab& operator=(const slab&) = delete;
    slab(slab&&) noexcept = delete;
    slab& operator=(slab&&) noexcept = delete;

    // returns: nullptr if failed, else the memory address of the block of memory
    // returns memory is properly aligned
    [[nodiscard]] void* alloc(size_t size);

    // returns: nullptr if failed, else the memory address of the block of memory
    // returns memory is properly aligned
    [[nodiscard]] void* calloc(size_t size);

    // NOT thread safe
    // returns: -1 if failed
    void reset();

    // returns: -1 if failed
    void free(void* ptr, size_t size);

    // returns true if freed successfully, false if not owned by this slab
    bool free_unsized(void* ptr);

    size_t get_pool_count() const;
    size_t get_total_capacity() const;
    size_t get_total_free() const;
    size_t get_pool_block_size(size_t index) const;
    size_t get_pool_free_space(size_t index) const;

    // check if pointer belongs to this slab
    bool owns(void* ptr) const;

    // get the contiguous memory region backing all pools
    std::byte* region_start() const { return m_region; }
    std::byte* region_end() const { return m_region + m_region_size; }

    static constexpr size_t size_to_index(size_t size)
    {
        if (size == 0 || size > Tconfig::SIZE_CLASS_CONFIG[Tconfig::NUM_SIZE_CLASSES - 1].byte_size)
            return static_cast<size_t>(-1);
        size_t s = size < Tconfig::SIZE_CLASS_CONFIG[0].byte_size ? Tconfig::SIZE_CLASS_CONFIG[0].byte_size : size;
        size_t vi = std::bit_width(std::bit_ceil(s)) - std::bit_width(Tconfig::SIZE_CLASS_CONFIG[0].byte_size);
        if (vi >= Tconfig::INDEX_SPAN)
            return static_cast<size_t>(-1);
        return Tconfig::INDEX_LUT[vi];
    }

    static constexpr size_t index_to_size_class(size_t index)
    {
        if (index >= Tconfig::NUM_SIZE_CLASSES)
            return 0;
        return Tconfig::SIZE_CLASS_CONFIG[index].byte_size;
    }

private:
    constexpr static size_t MAX_CACHED_SLABS = 4;

    struct cache_entry
    {
        size_t epoch;
        slab<Tconfig>* owner;
        std::array<thread_local_cache, Tconfig::NUM_CACHED_CLASSES> storage;

        void flush()
        {
            if (!owner)
                return; // should we assert?

            for (size_t i = 0; i < Tconfig::NUM_CACHED_CLASSES; i++)
            {
                // move pointers to appropriate pool from storage?
                auto& cache = storage[i];
                if (cache.is_empty())
                    continue;

                owner->shared_pools[i].free_batched_internal(cache.current, cache.objects.data());
                cache.current = 0;
            }
        }

        void invalidate_all()
        {
            if (!owner)
                return;

            for (size_t i = 0; i < Tconfig::NUM_CACHED_CLASSES; i++)
                storage[i].invalidate();
        }
    };

    inline thread_local static std::array<cache_entry, MAX_CACHED_SLABS> caches{};

    cache_entry* get_cached_slab()
    {
        assert(MAX_CACHED_SLABS != 0 && "Cannot get cached slab. Number of cached slabs is 0");

        // O(1) fast path: check the preferred hash slot first
        const size_t preferred = slab_id % MAX_CACHED_SLABS;
        if (caches[preferred].owner == this)
            return &caches[preferred];

        // Scan for an existing entry for this slab, or the first empty slot.
        // Slabs with colliding hash IDs will land in different slots when space is available.
        size_t empty_slot = caches[preferred].owner == nullptr ? preferred : (size_t)-1;
        for (size_t i = 0; i < MAX_CACHED_SLABS; ++i)
        {
            if (i == preferred)
                continue;
            if (caches[i].owner == this)
                return &caches[i];
            if (caches[i].owner == nullptr && empty_slot == (size_t)-1)
                empty_slot = i;
        }

        // Claim an empty slot (prefer the hash slot to keep affinity for next time)
        if (empty_slot != (size_t)-1)
        {
            cache_entry& entry = caches[empty_slot];
            entry.owner = this;
            entry.epoch = epoch.load(std::memory_order_acquire);
            init_cache_batch_sizes(entry);
            return &entry;
        }

        // All slots occupied by other slabs.
        // Evict the last slot (not the preferred one) so that slabs whose preferred
        // slots are 0..MAX_CACHED_SLABS-2 remain stable across round-robin cycling.
        // This mirrors LRU-ish eviction: the last slot acts as the "victim" slot.
        cache_entry& entry = caches[MAX_CACHED_SLABS - 1];
        entry.flush();
        entry.owner = this;
        entry.epoch = epoch.load(std::memory_order_acquire);
        init_cache_batch_sizes(entry);
        return &entry;
    }

    static void init_cache_batch_sizes(cache_entry& entry)
    {
        for (size_t i = 0; i < Tconfig::NUM_CACHED_CLASSES; ++i)
            entry.storage[i].batch_size = Tconfig::SIZE_CLASS_CONFIG[i].batch_size;
    }

    palloc_atomic<size_t> epoch;
    std::array<pool, Tconfig::NUM_SIZE_CLASSES> shared_pools;

    std::byte* m_region = nullptr;
    size_t m_region_size = 0; // total size of the contiguous region backing all pools in raw bytes

    inline static palloc_atomic<size_t> next_slab_id{0};
    size_t slab_id;
};

template<typename Tconfig>
slab<Tconfig>::slab() : epoch(0), slab_id(next_slab_id.fetch_add(1, std::memory_order_relaxed))
{
    constexpr size_t raw_size = Tconfig::compute_total_region_size();
    size_t page_size = AL::platform_mem::page_size();
    // m_region_size = ((raw_size + page_size - 1) / page_size) * page_size;
    m_region_size = VIRTUAL_MEM_ALLOC_SIZE; 

    void* mem = AL::platform_mem::virtual_alloc(m_region_size); // 100gb virtual memory alloc
    if (mem == nullptr)
        throw std::bad_alloc();

    m_region = static_cast<std::byte*>(mem);

    // carve sub-regions for each pool
    std::byte* cursor = m_region;
    for (size_t i = 0; i < Tconfig::NUM_SIZE_CLASSES; ++i)
    {
        auto& sc = Tconfig::SIZE_CLASS_CONFIG[i];
        // align cursor to block_size
        auto addr = reinterpret_cast<uintptr_t>(cursor);
        uintptr_t mask = sc.byte_size - 1;
        addr = (addr + mask) & ~mask;
        cursor = reinterpret_cast<std::byte*>(addr);

        shared_pools[i].init_from_region(cursor, sc.byte_size, sc.num_blocks);
        cursor += pool_view::required_region_size(sc.byte_size, sc.num_blocks);
    }
}

template<typename Tconfig>
slab<Tconfig>::~slab()
{
    // invalidate TLC entries for this slab
    const size_t preferred = slab_id % MAX_CACHED_SLABS;
    if (caches[preferred].owner == this)
    {
        caches[preferred].invalidate_all();
        caches[preferred].owner = nullptr;
    }
    else
    {
        for (size_t i = 0; i < MAX_CACHED_SLABS; ++i)
        {
            if (i == preferred)
                continue;
            if (caches[i].owner == this)
            {
                caches[i].invalidate_all();
                caches[i].owner = nullptr;
                break;
            }
        }
    }

    // munmap the single contiguous region (pools are non-owning, their destructors are no-ops)
    if (m_region != nullptr)
    {
        AL::platform_mem::free(m_region, m_region_size);
        m_region = nullptr;
    }
}

template<typename Tconfig>
void* slab<Tconfig>::alloc(size_t size)
{
    if (size == 0 || size == (size_t)-1) [[unlikely]]
        return nullptr;
    if (Tconfig::SIZE_CLASS_CONFIG[Tconfig::NUM_SIZE_CLASSES - 1].byte_size < size) [[unlikely]]
        return nullptr;

    size_t index = size_to_index(size);
    if (index == (size_t)-1) [[unlikely]]
        return nullptr;

    pool& p = shared_pools[index];

    if (index < Tconfig::NUM_CACHED_CLASSES) [[likely]]
    {
        auto cached_entry = get_cached_slab();
        thread_local_cache& cache = cached_entry->storage[index];
        size_t current_epoch = epoch.load(std::memory_order_acquire);
        if (cached_entry->epoch != current_epoch) [[unlikely]]
        {
            cached_entry->invalidate_all();
            cached_entry->epoch = current_epoch;
        }

        if (auto elem = cache.try_pop()) [[likely]]
            return elem;

        size_t num_allocated = p.alloc_batched_internal(cache.batch_size, cache.objects.data());
        cache.current = num_allocated;
        return cache.try_pop();
    }
    else
    {
        return p.alloc();
    }
}

template<typename Tconfig>
void* slab<Tconfig>::calloc(size_t size)
{
    void* ptr = alloc(size);
    if (ptr != nullptr)
    {
        size_t actual_size = Tconfig::SIZE_CLASS_CONFIG[size_to_index(size)].byte_size;
        std::memset(ptr, 0, actual_size);
    }
    return ptr;
}

template<typename Tconfig>
void slab<Tconfig>::reset()
{
    for (auto& p : shared_pools)
        p.reset();
    epoch.fetch_add(1, std::memory_order_release);
}

template<typename Tconfig>
void slab<Tconfig>::free(void* ptr, size_t size)
{
    if (size == 0 || size == (size_t)-1) [[unlikely]]
        return;
    if (Tconfig::SIZE_CLASS_CONFIG[Tconfig::NUM_SIZE_CLASSES - 1].byte_size < size) [[unlikely]]
        return;

    size_t index = size_to_index(size);
    if (index == (size_t)-1) [[unlikely]]
        return;

    pool& p = shared_pools[index];
    if (index < Tconfig::NUM_CACHED_CLASSES) [[likely]]
    {
        auto cached_entry = get_cached_slab();
        thread_local_cache& cache = cached_entry->storage[index];
        size_t current_epoch = epoch.load(std::memory_order_acquire);
        if (cached_entry->epoch != current_epoch) [[unlikely]]
        {
            cached_entry->invalidate_all();
            cached_entry->epoch = current_epoch;
        }

        if (cache.is_full()) [[unlikely]]
        {
            p.free_batched_internal(cache.batch_size, cache.objects.data() + (cache.current - cache.batch_size));
            cache.current -= cache.batch_size;
        }
        cache.push(ptr);
    }
    else
    {
        shared_pools[index].free(ptr);
    }
}

template<typename Tconfig>
bool slab<Tconfig>::free_unsized(void* ptr)
{
    for (size_t i = 0; i < Tconfig::NUM_SIZE_CLASSES; ++i)
    {
        pool& p = shared_pools[i];
        if (p.owns(ptr))
        {
            if (i < Tconfig::NUM_CACHED_CLASSES) [[likely]]
            {
                auto cached_entry = get_cached_slab();
                thread_local_cache& cache = cached_entry->storage[i];
                size_t current_epoch = epoch.load(std::memory_order_acquire);
                if (cached_entry->epoch != current_epoch) [[unlikely]]
                {
                    cached_entry->invalidate_all();
                    cached_entry->epoch = current_epoch;
                }

                if (cache.is_full()) [[unlikely]]
                {
                    p.free_batched_internal(cache.batch_size, cache.objects.data() + (cache.current - cache.batch_size));
                    cache.current -= cache.batch_size;
                }
                cache.push(ptr);
            }
            else
            {
                p.free(ptr);
            }
            return true;
        }
    }
    return false;
}

template<typename Tconfig>
size_t slab<Tconfig>::get_pool_count() const
{
    return std::size(shared_pools);
}

template<typename Tconfig>
size_t slab<Tconfig>::get_total_capacity() const
{
    size_t total = 0;
    for (const auto& p : shared_pools)
        total += p.get_capacity();
    return total;
}

template<typename Tconfig>
size_t slab<Tconfig>::get_total_free() const
{
    size_t total = 0;
    for (const auto& p : shared_pools)
        total += p.get_free_space();
    return total;
}

template<typename Tconfig>
size_t slab<Tconfig>::get_pool_block_size(size_t index) const
{
    if (index >= Tconfig::NUM_SIZE_CLASSES)
        return 0;
    return shared_pools[index].get_block_size();
}

template<typename Tconfig>
size_t slab<Tconfig>::get_pool_free_space(size_t index) const
{
    if (index >= Tconfig::NUM_SIZE_CLASSES)
        return 0;
    return shared_pools[index].get_free_space();
}

template<typename Tconfig>
bool slab<Tconfig>::owns(void* ptr) const
{
    for (const auto& p : shared_pools)
        if (p.owns(ptr))
            return true;
    return false;
}

using default_slab = slab<slab_config<>>;

} // namespace AL
