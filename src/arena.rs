//! An arena in one reserved range of address space. Node `id` lives at
//! offset `id * size_of::<T>()`, ids are handed out in order, and a node never
//! moves once pushed. The kernel backs a page of the range with memory only
//! once a node on it is written, and takes it back when the arena releases it.

use std::mem::{size_of, MaybeUninit};
use std::ops::{Index, IndexMut};

/// Nodes the arena reserves room for; ids stay below `2^31`.
const MAX_NODES: usize = 1 << 31;
/// Fewest nodes a reservation may shrink to when the kernel refuses more.
const MIN_NODES: usize = 1 << 24;
/// Bytes at the start of the range that stay resident across `truncate`, so
/// that the many small declarations of a library neither call the kernel nor
/// fault pages back in.
const KEEP_BYTES: usize = 32 << 20;

pub(crate) struct Arena<T> {
    base: *mut MaybeUninit<T>,
    cap: usize,
    len: usize,
    /// One past the highest node written since the range was last released.
    high: usize,
    page: usize,
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
                return Arena { base: p.cast(), cap, len: 0, high: 0, page };
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
        // SAFETY: `i < cap` lies inside the reserved range.
        unsafe { (*self.base.add(i)).write(x) };
        self.len = i + 1;
        self.high = self.high.max(self.len);
    }

    /// Keep the first `n` nodes and give the pages past them back to the
    /// kernel, except the first `KEEP_BYTES` of the range.
    pub(crate) fn truncate(&mut self, n: usize) {
        if std::mem::needs_drop::<T>() {
            for i in n..self.len {
                // SAFETY: slot `i < len` was written by `push` and is dropped once.
                unsafe { (*self.base.add(i)).assume_init_drop() };
            }
        }
        self.len = self.len.min(n);
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
        assert!(i < self.len, "arena index {i} out of range {}", self.len);
        // SAFETY: every slot below `len` was written by `push`.
        unsafe { (*self.base.add(i)).assume_init_ref() }
    }
}

impl<T> IndexMut<usize> for Arena<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        assert!(i < self.len, "arena index {i} out of range {}", self.len);
        // SAFETY: as in `index`; `&mut self` makes the access exclusive.
        unsafe { (*self.base.add(i)).assume_init_mut() }
    }
}
