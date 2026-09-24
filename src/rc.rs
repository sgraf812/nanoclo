//! Reference counts for the evaluator's values, spines and environments.
//!
//! A node's count is the number of its holders: the nodes that name it as a
//! child, the forced cell or domain that records it, the caches that keep it,
//! and the handles (`V`, `S`, `E`) held by the evaluator. Nodes and weak
//! tables name each other by raw id; a handle is a raw id that holds a
//! reference, taken by `own` and given back when it drops.
//!
//! A node whose count a handle brings to zero is not freed at once: it
//! becomes a zombie and enters the ring, and an intern hit on it makes it
//! live again. `nb_collect` frees the zombies that have been in the ring for
//! `RING` deaths. Freeing a node releases its children; a child whose count
//! that brings to zero is freed as well, unless a handle released it within
//! the last `RING` deaths, in which case it becomes a zombie of its own.
//!
//! Each count shares a 32-bit word with a stamp: `ALLOC | stamp << 12 | count`.
//! The count saturates at `COUNT_MAX`, and a saturated node lives until the
//! end of the declaration. The stamp is the ring clock at the node's last
//! release by a handle. A word of zero marks a slot with no node.
//!
//! The counts live apart from the nodes, in a store that only this module
//! touches, so a handle can update its count while the evaluator holds the
//! arenas mutably. Each checker context installs its own store for as long as
//! it lives; a context created while another one is alive stacks on top.

use crate::arena::Arena;
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;

const ALLOC: u32 = 1 << 31;
const COUNT_BITS: u32 = 12;
const COUNT_MAX: u32 = (1 << COUNT_BITS) - 1;
const STAMP_BITS: u32 = 19;
const STAMP_MASK: u32 = (1 << STAMP_BITS) - 1;
/// Deaths a zombie waits in the ring before it is freed.
/// TEMP experiment: `NANOCLO_RING` sets it as a power of two.
fn ring_cap() -> usize {
    static R: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *R.get_or_init(|| 1usize << std::env::var("NANOCLO_RING").ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(18))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    Val,
    Spine,
    Env,
}

pub(crate) struct Counts {
    pub(crate) vals: Arena<u32>,
    pub(crate) spines: Arena<u32>,
    pub(crate) envs: Arena<u32>,
    /// Zombies in order of death, with their stamps.
    ring: VecDeque<(Kind, u32, u32)>,
    /// Deaths so far, modulo `2^STAMP_BITS`.
    clock: u32,
}

impl Counts {
    #[inline]
    fn of(&mut self, k: Kind) -> &mut Arena<u32> {
        match k {
            Kind::Val => &mut self.vals,
            Kind::Spine => &mut self.spines,
            Kind::Env => &mut self.envs,
        }
    }
}

thread_local! {
    /// References taken by `pin` and never given back, for the consistency
    /// check of `check_counts`.
    pub(crate) static PINS: Cell<u64> = const { Cell::new(0) };
    static STORES: RefCell<Vec<Box<Counts>>> = const { RefCell::new(Vec::new()) };
    static CURRENT: Cell<*mut Counts> = const { Cell::new(std::ptr::null_mut()) };
}

/// Installs a fresh store for one checker context and removes it on drop.
pub(crate) struct CountsGuard(());

impl CountsGuard {
    pub(crate) fn new() -> Self {
        let mut c = Box::new(Counts {
            vals: Arena::new(),
            spines: Arena::new(),
            envs: Arena::new(),
            ring: VecDeque::new(),
            clock: 0,
        });
        CURRENT.with(|cur| cur.set(&mut *c));
        STORES.with(|s| s.borrow_mut().push(c));
        CountsGuard(())
    }
}

impl Drop for CountsGuard {
    fn drop(&mut self) {
        STORES.with(|s| {
            let mut s = s.borrow_mut();
            s.pop();
            let next = s.last_mut().map_or(std::ptr::null_mut(), |c| &mut **c as *mut Counts);
            CURRENT.with(|cur| cur.set(next));
        });
    }
}

#[inline]
pub(crate) fn counts() -> &'static mut Counts {
    // SAFETY: the store is boxed, so it does not move while it is installed,
    // and only this module reaches it, one access at a time on the owning
    // thread.
    unsafe { &mut *CURRENT.with(|cur| cur.get()) }
}

macro_rules! handle {
    ($name:ident, $kind:expr, $doc:literal) => {
        #[doc = $doc]
        pub(crate) struct $name(u32);

        impl $name {
            /// Take a reference to node `id`.
            #[inline]
            pub(crate) fn own(id: u32) -> Self {
                inc($kind, id);
                $name(id)
            }

            #[inline]
            pub(crate) fn id(&self) -> u32 { self.0 }

            /// Keep the reference for the rest of the declaration: for an id
            /// handed to the closure checker, which does not count.
            #[inline]
            #[allow(dead_code)]
            pub(crate) fn pin(self) -> u32 {
                let id = self.0;
                std::mem::forget(self);
                PINS.with(|p| p.set(p.get() + 1));
                id
            }
        }

        impl Clone for $name {
            #[inline]
            fn clone(&self) -> Self { Self::own(self.0) }
        }

        impl Drop for $name {
            #[inline]
            fn drop(&mut self) { release($kind, self.0) }
        }

        impl PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
        }
        impl Eq for $name {}
    };
}

handle!(V, Kind::Val, "A value held by the evaluator.");
handle!(S, Kind::Spine, "A spine held by the evaluator.");
handle!(E, Kind::Env, "An environment held by the evaluator.");

/// The word of a slot holding a node with `count` holders.
#[inline]
fn word(count: u32) -> u32 { ALLOC | count }

/// Record a new node of kind `k`, with no holders yet.
#[inline]
pub(crate) fn new_node(k: Kind) { counts().of(k).push(word(0)) }

/// Record a sentinel of kind `k`, held for as long as the context lives.
pub(crate) fn new_sentinel(k: Kind) { counts().of(k).push(word(1)) }

/// Forget every node of kind `k` past the first `n`, which are sentinels, and
/// the ring and pins with them.
pub(crate) fn reset(k: Kind, n: usize) {
    let c = counts();
    c.of(k).truncate(n);
    for i in 0..n {
        c.of(k)[i] = word(1);
    }
    c.ring.clear();
    c.clock = 0;
    PINS.with(|p| p.set(0));
}

/// Whether node `id` of kind `k` is live or a zombie.
#[inline]
pub(crate) fn alive(k: Kind, id: u32) -> bool {
    let a = counts().of(k);
    a.present(id as usize) && a[id as usize] & ALLOC != 0
}

/// Take a reference to node `id`; a zombie becomes live again.
#[inline]
pub(crate) fn inc(k: Kind, id: u32) {
    let w = &mut counts().of(k)[id as usize];
    assert!(*w & ALLOC != 0, "{k:?} {id}: reference to a freed node");
    if *w & COUNT_MAX != COUNT_MAX {
        *w += 1;
    }
}

/// Give back a handle's reference to node `id`, recording the release in its
/// stamp. A node this leaves without holders becomes a zombie.
#[inline]
pub(crate) fn release(k: Kind, id: u32) {
    let c = counts();
    let clock = c.clock;
    let w = &mut c.of(k)[id as usize];
    let count = *w & COUNT_MAX;
    if count == COUNT_MAX {
        return;
    }
    assert!(count > 0 && *w & ALLOC != 0, "{k:?} {id}: release without a reference");
    *w = ALLOC | clock << COUNT_BITS | (count - 1);
    if count == 1 {
        c.ring.push_back((k, id, clock));
        c.clock = (clock + 1) & STAMP_MASK;
    }
}

/// Give back the reference a freed node held on `id`. Returns the stamp when
/// this leaves `id` without holders.
#[inline]
fn drop_child(c: &mut Counts, k: Kind, id: u32) -> Option<u32> {
    let w = &mut c.of(k)[id as usize];
    let count = *w & COUNT_MAX;
    if count == COUNT_MAX {
        return None;
    }
    assert!(count > 0, "{k:?} {id}: freed parent held no reference");
    *w -= 1;
    (count == 1).then_some((*w >> COUNT_BITS) & STAMP_MASK)
}

/// Take a reference to value `id`.
#[inline]
pub(crate) fn inc_val(id: u32) { inc(Kind::Val, id) }
/// Take a reference to spine `id`.
#[inline]
pub(crate) fn inc_spine(id: u32) { inc(Kind::Spine, id) }
/// Take a reference to environment `id`.
#[inline]
pub(crate) fn inc_env(id: u32) { inc(Kind::Env, id) }
/// Pin environment `id` for the rest of the declaration.
#[inline]
pub(crate) fn pin_env(id: u32) {
    inc_env(id);
    PINS.with(|p| p.set(p.get() + 1));
}
/// Give back a reference to value `id`.
#[inline]
pub(crate) fn dec_val(id: u32) { release(Kind::Val, id) }

/// Whether the ring holds zombies that are due.
#[inline]
pub(crate) fn collect_due() -> bool { counts().ring.len() > ring_cap() }

impl<'x, 't: 'x, 'p: 't> crate::tc::TypeChecker<'x, 't, 'p> {
    /// Free the zombies that have waited `RING` deaths, and every node that
    /// freeing them leaves without holders and without a recent release.
    pub(crate) fn nb_collect(&mut self) {
        use crate::closure::{Entry, VIEW_BIT};
        use crate::nbe::{Elim, RigidHead, Value};
        let c = counts();
        let mut work: Vec<(Kind, u32)> = Vec::new();
        let mut children: Vec<(Kind, u32)> = Vec::new();
        while c.ring.len() > ring_cap() {
            let (k, id, stamp) = c.ring.pop_front().unwrap();
            let a = c.of(k);
            if !a.present(id as usize) || a[id as usize] != (ALLOC | stamp << COUNT_BITS) {
                continue;
            }
            work.push((k, id));
            while let Some((k, id)) = work.pop() {
                children.clear();
                match k {
                    Kind::Val => match self.ctx.nb.vals[id as usize] {
                        Value::Rigid { head, spine } => {
                            if let RigidHead::BVar(_, ty) = head {
                                children.push((Kind::Val, ty));
                            }
                            children.push((Kind::Spine, spine));
                        }
                        Value::Unfold { spine, forced, .. } => {
                            children.push((Kind::Spine, spine));
                            children.extend(forced.filter(|&f| f != id).map(|f| (Kind::Val, f)));
                        }
                        Value::Lam { domain, env, .. } => {
                            children.push((Kind::Env, env));
                            children.extend(domain.map(|d| (Kind::Val, d)));
                        }
                        Value::Pi { domain, env, .. } => {
                            children.push((Kind::Val, domain));
                            children.push((Kind::Env, env));
                        }
                        Value::Thunk { env, forced, .. } => {
                            children.push((Kind::Env, env));
                            children.extend(forced.filter(|&f| f != id).map(|f| (Kind::Val, f)));
                        }
                        Value::Sort { .. } | Value::NatLit { .. } | Value::StrLit { .. } => {}
                    },
                    Kind::Spine => {
                        let n = &self.ctx.nb.spines[id as usize];
                        children.push((Kind::Spine, n.parent));
                        if let Elim::App(a) = n.elim {
                            children.push((Kind::Val, a));
                        }
                    }
                    Kind::Env => {
                        let n = &self.ctx.rp.envs[id as usize];
                        children.push((Kind::Env, n.parent));
                        match n.entry {
                            Entry::V(v) => children.push((Kind::Val, v)),
                            Entry::Val(_, env) if env & VIEW_BIT == 0 => children.push((Kind::Env, env)),
                            _ => {}
                        }
                    }
                }
                c.of(k)[id as usize] = 0;
                c.of(k).free(id as usize);
                match k {
                    Kind::Val => self.ctx.nb.vals.free(id as usize),
                    Kind::Spine => self.ctx.nb.spines.free(id as usize),
                    Kind::Env => self.ctx.rp.envs.free(id as usize),
                }
                for &(ck, cid) in &children {
                    let Some(stamp) = drop_child(c, ck, cid) else { continue };
                    if (c.clock.wrapping_sub(stamp) & STAMP_MASK) < ring_cap() as u32 {
                        c.ring.push_back((ck, cid, stamp));
                    } else {
                        work.push((ck, cid));
                    }
                }
            }
        }
    }
}

impl<'t, 'p> crate::util::TcCtx<'t, 'p> {
    /// Recompute every count from the live nodes and the caches that hold
    /// references, and panic unless each stored count covers its expected
    /// one and the surplus of the unsaturated counts is at most the pins. Run
    /// when no handle is alive.
    pub(crate) fn check_counts(&self) {
        use crate::closure::{Entry, VIEW_BIT};
        use crate::nbe::{Elim, RigidHead, Value};
        let c = counts();
        let live = |a: &Arena<u32>, i: usize| a.present(i) && a[i] & ALLOC != 0;
        let mut ev = vec![0u64; self.nb.vals.len()];
        let mut es = vec![0u64; self.nb.spines.len()];
        let mut ee = vec![0u64; self.rp.envs.len()];
        es[0] += 1;
        ee[0] += 1;
        for i in (0..self.nb.vals.len()).filter(|&i| live(&c.vals, i)) {
            match self.nb.vals[i] {
                Value::Rigid { head, spine } => {
                    if let RigidHead::BVar(_, ty) = head {
                        ev[ty as usize] += 1;
                    }
                    es[spine as usize] += 1;
                }
                Value::Unfold { spine, forced, .. } => {
                    es[spine as usize] += 1;
                    if let Some(f) = forced.filter(|&f| f as usize != i) {
                        ev[f as usize] += 1;
                    }
                }
                Value::Lam { domain, env, .. } => {
                    ee[env as usize] += 1;
                    if let Some(d) = domain {
                        ev[d as usize] += 1;
                    }
                }
                Value::Pi { domain, env, .. } => {
                    ev[domain as usize] += 1;
                    ee[env as usize] += 1;
                }
                Value::Thunk { env, forced, .. } => {
                    ee[env as usize] += 1;
                    if let Some(f) = forced.filter(|&f| f as usize != i) {
                        ev[f as usize] += 1;
                    }
                }
                Value::Sort { .. } | Value::NatLit { .. } | Value::StrLit { .. } => {}
            }
        }
        for i in (1..self.nb.spines.len()).filter(|&i| live(&c.spines, i)) {
            let n = &self.nb.spines[i];
            es[n.parent as usize] += 1;
            if let Elim::App(a) = n.elim {
                ev[a as usize] += 1;
            }
        }
        for i in (1..self.rp.envs.len()).filter(|&i| live(&c.envs, i)) {
            let n = &self.rp.envs[i];
            ee[n.parent as usize] += 1;
            match n.entry {
                Entry::V(v) => ev[v as usize] += 1,
                Entry::Val(_, env) if env & VIEW_BIT == 0 => ee[env as usize] += 1,
                _ => {}
            }
        }
        let nb = &self.nb;
        for v in nb.clo_val_cache.values().chain(nb.const_val_cache.values()).chain(nb.const_ty_cache.values())
            .chain(nb.rec_rule_cache.values()).chain(nb.local_cache.values()).chain(nb.unfold_cache.values().flatten())
        {
            ev[v.id() as usize] += 1;
        }
        let mut surplus = 0u64;
        for (what, stored, expected) in [("value", &c.vals, &ev), ("spine", &c.spines, &es), ("env", &c.envs, &ee)] {
            for (i, &e) in expected.iter().enumerate() {
                if !live(stored, i) {
                    assert!(e == 0, "{what} {i}: freed with {e} live holders");
                    continue;
                }
                let s = u64::from(stored[i] & COUNT_MAX);
                if s == u64::from(COUNT_MAX) {
                    continue;
                }
                assert!(s >= e, "{what} {i}: count {s} below its {e} holders");
                surplus += s - e;
            }
        }
        let pins = PINS.with(|p| p.get());
        assert!(surplus <= pins, "counts exceed their holders by {surplus}, but only {pins} references are pinned");
    }
}

/// Make room in a full weak table by dropping the entries `keep` rejects,
/// and grow it now if that frees less than half of it, so that the next pass
/// comes only after as many inserts again.
pub(crate) fn make_room<K, W, B>(m: &mut std::collections::HashMap<K, W, B>, keep: impl FnMut(&K, &mut W) -> bool)
where
    K: std::hash::Hash + Eq,
    B: std::hash::BuildHasher,
{
    if m.len() == m.capacity() {
        m.retain(keep);
        if m.len() * 2 > m.capacity() {
            m.reserve(m.capacity());
        }
    }
}

/// `make_room` for a weak set.
pub(crate) fn make_room_set<K, B>(m: &mut std::collections::HashSet<K, B>, keep: impl FnMut(&K) -> bool)
where
    K: std::hash::Hash + Eq,
    B: std::hash::BuildHasher,
{
    if m.len() == m.capacity() {
        m.retain(keep);
        if m.len() * 2 > m.capacity() {
            m.reserve(m.capacity());
        }
    }
}

/// The id interned under `key`, or `mk()` interned under it when there is
/// none or the one there has been freed.
#[inline]
pub(crate) fn intern<K, B>(
    table: &mut std::collections::HashMap<K, u32, B>,
    k: Kind,
    key: K,
    mk: impl FnOnce() -> u32,
) -> u32
where
    K: std::hash::Hash + Eq,
    B: std::hash::BuildHasher,
{
    make_room(table, |_, v| alive(k, *v));
    match table.entry(key) {
        std::collections::hash_map::Entry::Occupied(mut o) => {
            if alive(k, *o.get()) {
                *o.get()
            } else {
                let id = mk();
                *o.get_mut() = id;
                id
            }
        }
        std::collections::hash_map::Entry::Vacant(v) => *v.insert(mk()),
    }
}
