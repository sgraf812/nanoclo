//! An arena in one reserved range of address space. Node `id` lives at
//! offset `id * size_of::<T>()`, ids are handed out in order, and a node never
//! moves once pushed. The kernel backs a page of the range with memory only
//! once a node on it is written, and takes it back when the arena releases it.
//!
//! Nodes are freed one at a time but given back to the kernel a block of
//! `BLOCK` nodes at a time: a block is released once every slot in it has been
//! used and every node in it freed. A freed node keeps its bytes until its
//! block goes, and ids are never reused, so reading a freed node still reads
//! the term it held. A released block reads as zeros; every access checks
//! that its block is still there and panics otherwise.

use std::mem::{needs_drop, size_of, MaybeUninit};
use std::ops::{Index, IndexMut};

/// Nodes the arena reserves room for; ids stay below `2^31`.
const MAX_NODES: usize = 1 << 31;
/// Fewest nodes a reservation may shrink to when the kernel refuses more.
const MIN_NODES: usize = 1 << 24;
/// Bytes at the start of the range that stay resident across `truncate`, so
/// that the many small declarations of a library neither call the kernel nor
/// fault pages back in.
const KEEP_BYTES: usize = 32 << 20;
/// Nodes per block, the unit in which freed nodes go back to the kernel. A
/// block of any node size spans whole pages of 4096 bytes.
const BLOCK_BITS: usize = 12;
const BLOCK: usize = 1 << BLOCK_BITS;

pub(crate) struct Arena<T> {
    base: *mut MaybeUninit<T>,
    cap: usize,
    len: usize,
    /// One past the highest node written since the range was last released.
    high: usize,
    page: usize,
    /// Nodes of each block that have been pushed and not freed.
    live: Vec<u32>,
    /// Whether each block has been given back to the kernel.
    released: Vec<bool>,
}

// SAFETY: the arena owns its range exclusively, like a `Vec`.
unsafe impl<T: Send> Send for Arena<T> {}

impl<T> Arena<T> {
    pub(crate) fn new() -> Self {
        let mut cap = MAX_NODES;
        loop {
            // SAFETY: an anonymous private mapping with no reservation of swap;
            // the result is checked below.
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    cap * size_of::<T>(),
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            if p != libc::MAP_FAILED {
                // SAFETY: `sysconf` has no preconditions.
                let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap_or(4096);
                return Arena { base: p.cast(), cap, len: 0, high: 0, page, live: Vec::new(), released: Vec::new() };
            }
            assert!(cap > MIN_NODES, "cannot reserve address space for an arena of {MIN_NODES} nodes");
            cap /= 2;
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize { self.len }

    #[inline]
    pub(crate) fn push(&mut self, x: T) {
        let i = self.len;
        assert!(i < self.cap, "arena exhausted at {i} nodes");
        if i % BLOCK == 0 {
            // the previous block is now full, and may be empty already
            if let Some(b) = (i >> BLOCK_BITS).checked_sub(1) {
                if self.live[b] == 0 {
                    self.release_block(b);
                }
            }
            self.live.push(0);
            self.released.push(false);
        }
        self.live[i >> BLOCK_BITS] += 1;
        // SAFETY: `i < cap` lies inside the reserved range.
        unsafe { (*self.base.add(i)).write(x) };
        self.len = i + 1;
        self.high = self.high.max(self.len);
    }

    /// Record that node `i` is dead. Its block goes back to the kernel once
    /// the block is full and all its nodes are dead.
    pub(crate) fn free(&mut self, i: usize) {
        assert!(i < self.len && !self.released[i >> BLOCK_BITS], "freeing a released node {i}");
        let b = i >> BLOCK_BITS;
        self.live[b] -= 1;
        if self.live[b] == 0 && (b + 1) * BLOCK <= self.len {
            self.release_block(b);
        }
    }

    fn release_block(&mut self, b: usize) {
        if needs_drop::<T>() {
            unreachable!("released blocks hold only nodes without drop code");
        }
        let bytes = BLOCK * size_of::<T>();
        // SAFETY: block `b` lies inside the reserved range, is page-aligned
        // since `bytes` is a multiple of 4096, and holds no live node; the
        // kernel zero-fills it if it is touched again.
        unsafe { libc::madvise(self.base.cast::<u8>().add(b * bytes).cast(), bytes, libc::MADV_DONTNEED) };
        self.released[b] = true;
    }

    /// The id ranges of the blocks still present.
    pub(crate) fn present_ranges(&self) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        (0..self.released.len())
            .filter(|&b| !self.released[b])
            .map(|b| b * BLOCK..((b + 1) * BLOCK).min(self.len))
    }

    /// Whether node `i` still has its block.
    #[inline]
    pub(crate) fn present(&self, i: usize) -> bool { i < self.len && !self.released[i >> BLOCK_BITS] }

    /// Keep the first `n` nodes and give the pages past them back to the
    /// kernel, except the first `KEEP_BYTES` of the range.
    pub(crate) fn truncate(&mut self, n: usize) {
        if needs_drop::<T>() {
            for i in n..self.len {
                // SAFETY: slot `i < len` was written by `push` and is dropped once.
                unsafe { (*self.base.add(i)).assume_init_drop() };
            }
        }
        self.len = self.len.min(n);
        let blocks = n.div_ceil(BLOCK);
        self.live.truncate(blocks);
        self.released.truncate(blocks);
        if let Some(last) = self.live.last_mut() {
            *last = u32::try_from(n - (blocks - 1) * BLOCK).unwrap();
        }
        if let Some(b) = self.released.iter().position(|&r| r) {
            unreachable!("block {b} released below a truncation to {n}");
        }
        let from = (n * size_of::<T>()).max(KEEP_BYTES).next_multiple_of(self.page);
        let to = self.high * size_of::<T>();
        if to > from {
            // SAFETY: `[from, to)` lies inside the reserved range and holds no
            // live node; the kernel zero-fills it when it is touched again.
            unsafe { libc::madvise(self.base.cast::<u8>().add(from).cast(), to - from, libc::MADV_DONTNEED) };
            self.high = from / size_of::<T>();
        }
    }
}

impl<T> Drop for Arena<T> {
    fn drop(&mut self) {
        self.truncate(0);
        // SAFETY: the range was mapped in `new` with this length.
        unsafe { libc::munmap(self.base.cast(), self.cap * size_of::<T>()) };
    }
}

impl<T> Index<usize> for Arena<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        assert!(self.present(i), "arena index {i} out of range {} or released", self.len);
        // SAFETY: every slot below `len` was written by `push`.
        unsafe { (*self.base.add(i)).assume_init_ref() }
    }
}

impl<T> IndexMut<usize> for Arena<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        assert!(self.present(i), "arena index {i} out of range {} or released", self.len);
        // SAFETY: as in `index`; `&mut self` makes the access exclusive.
        unsafe { (*self.base.add(i)).assume_init_mut() }
    }
}
