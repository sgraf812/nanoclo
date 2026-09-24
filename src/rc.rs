//! Reference counts for the evaluator's values, spines and environments.
//!
//! A node's count is the number of its holders: the nodes that name it as a
//! child, the forced cell or domain that records it, the caches that keep it,
//! and the handles (`V`, `S`, `E`) held by the evaluator. Nodes and weak
//! tables name each other by raw id; a handle is a raw id that holds a
//! reference, taken by `own` and given back when it drops.
//!
//! A node whose count a handle brings to zero is not freed at once: it
//! becomes a zombie, and an intern hit on it makes it live again. Each count
//! shares a 32-bit word with a stamp, `ALLOC | stamp << 12 | count`, where the
//! stamp is the epoch of the node's last release by a handle. `nb_collect`
//! starts a new epoch and sweeps the count words: it frees every zombie
//! released before the previous epoch, and releases the children of each.
//! A child that this leaves without holders is freed as well, unless a
//! handle released it in the current or the previous epoch.
//!
//! The count saturates at `COUNT_MAX`, and a saturated node lives until the
//! end of the declaration. A word of zero marks a slot with no node.
//!
//! A count word lives in its node's slot (`arena.rs`). A handle updates it
//! through a raw pointer to the word alone, computed from the arena's base,
//! so it never overlaps a reference the evaluator holds into a node. Each
//! checker context registers its arenas' bases here for as long as it lives;
//! a context created while another one is alive stacks on top.

use crate::arena::Arena;
use std::cell::{Cell, RefCell};

/// The word of a new node, which has no holders yet.
pub(crate) const FRESH: u32 = ALLOC;
/// The word of a sentinel, held for as long as its context lives.
pub(crate) const SENTINEL: u32 = ALLOC | 1;

const ALLOC: u32 = 1 << 31;
const COUNT_BITS: u32 = 12;
const COUNT_MAX: u32 = (1 << COUNT_BITS) - 1;
const STAMP_BITS: u32 = 19;
const STAMP_MASK: u32 = (1 << STAMP_BITS) - 1;
/// Fewest allocations between two sweeps.
const SWEEP_MIN: u64 = 1 << 20;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    Val,
    Spine,
    Env,
}

pub(crate) struct Counts {
    /// For each kind, node 0's word and the distance between two words.
    words: [(*mut u32, usize); 3],
    /// Sweeps so far, modulo `2^STAMP_BITS`.
    epoch: u32,
    /// Allocations since the last sweep, and how many the next one waits for.
    allocs: u64,
    next_sweep: u64,
}

impl Counts {
    /// The count word of node `id` of kind `k`. A released block reads as
    /// zeros, so its words show no `ALLOC`.
    #[inline]
    fn word(&self, k: Kind, id: u32) -> *mut u32 {
        let (base, stride) = self.words[k as usize];
        assert!(!base.is_null(), "{k:?} arena not registered");
        // SAFETY: ids come from the arena's pushes, so the word lies inside
        // its reserved range.
        unsafe { base.cast::<u8>().add(id as usize * stride).cast() }
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
            words: [(std::ptr::null_mut(), 0); 3],
            epoch: 0,
            allocs: 0,
            next_sweep: SWEEP_MIN,
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

/// Make the counts of arena `a` reachable for kind `k`.
pub(crate) fn register<T>(k: Kind, a: &Arena<T>) { counts().words[k as usize] = a.words() }

/// Count one allocation towards the next sweep.
#[inline]
pub(crate) fn new_node() { counts().allocs += 1 }

/// Forget the epochs and pins of the declaration that ended.
pub(crate) fn reset() {
    let c = counts();
    c.epoch = 0;
    c.allocs = 0;
    c.next_sweep = SWEEP_MIN;
    PINS.with(|p| p.set(0));
}

/// Whether node `id` of kind `k` is live or a zombie.
#[inline]
pub(crate) fn alive(k: Kind, id: u32) -> bool {
    // SAFETY: see `Counts::word`.
    unsafe { *counts().word(k, id) & ALLOC != 0 }
}

/// Take a reference to node `id`; a zombie becomes live again.
#[inline]
pub(crate) fn inc(k: Kind, id: u32) {
    // SAFETY: see `Counts::word`; no reference to the word exists.
    let w = unsafe { &mut *counts().word(k, id) };
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
    let epoch = c.epoch;
    // SAFETY: as in `inc`.
    let w = unsafe { &mut *c.word(k, id) };
    let count = *w & COUNT_MAX;
    if count == COUNT_MAX {
        return;
    }
    assert!(count > 0 && *w & ALLOC != 0, "{k:?} {id}: release without a reference");
    *w = ALLOC | epoch << COUNT_BITS | (count - 1);
}

/// Give back the reference a freed node held on `id`. Returns the stamp when
/// this leaves `id` without holders.
#[inline]
fn drop_child(c: &mut Counts, k: Kind, id: u32) -> Option<u32> {
    // SAFETY: as in `inc`.
    let w = unsafe { &mut *c.word(k, id) };
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

/// Whether enough has been allocated since the last sweep.
#[inline]
pub(crate) fn collect_due() -> bool {
    let c = counts();
    c.allocs >= c.next_sweep
}

impl<'x, 't: 'x, 'p: 't> crate::tc::TypeChecker<'x, 't, 'p> {
    /// Start a new epoch and free every zombie released before the previous
    /// one, and every node that freeing them leaves without holders and
    /// without a release in the current or the previous epoch.
    pub(crate) fn nb_collect(&mut self) {
        use crate::closure::{Entry, VIEW_BIT};
        use crate::nbe::{Elim, RigidHead, Value};
        let c = counts();
        c.epoch = (c.epoch + 1) & STAMP_MASK;
        c.allocs = 0;
        let epoch = c.epoch;
        let stale = |stamp: u32| (epoch.wrapping_sub(stamp) & STAMP_MASK) >= 2;
        let mut work: Vec<(Kind, u32)> = Vec::new();
        let mut children: Vec<(Kind, u32)> = Vec::new();
        let mut zombies: Vec<(Kind, u32)> = Vec::new();
        fn scan<T>(a: &Arena<T>, k: Kind, stale: &impl Fn(u32) -> bool, out: &mut Vec<(Kind, u32)>) {
            for range in a.present_ranges() {
                for i in range {
                    let w = a.word(i);
                    if w & ALLOC != 0 && w & COUNT_MAX == 0 && stale((w >> COUNT_BITS) & STAMP_MASK) {
                        out.push((k, i as u32));
                    }
                }
            }
        }
        scan(&self.ctx.nb.vals, Kind::Val, &stale, &mut zombies);
        scan(&self.ctx.nb.spines, Kind::Spine, &stale, &mut zombies);
        scan(&self.ctx.rp.envs, Kind::Env, &stale, &mut zombies);
        for (k, id) in zombies {
            // an earlier cascade in this sweep may have freed it already
            // SAFETY: see `Counts::word`.
            let w = unsafe { *c.word(k, id) };
            if w & ALLOC == 0 || w & COUNT_MAX != 0 {
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
                if k == Kind::Val {
                    // the reduct and the type recorded for it are its children too
                    if let Some(Some(r)) = self.ctx.nb.iota_cache.remove(&id) {
                        if r != id {
                            children.push((Kind::Val, r));
                        }
                    }
                    if let Some(t) = self.ctx.nb.type_cache.remove(&id) {
                        if t != id {
                            children.push((Kind::Val, t));
                        }
                    }
                }
                let i = id as usize;
                match k {
                    Kind::Val => {
                        self.ctx.nb.vals.set_word(i, 0);
                        self.ctx.nb.vals.free(i);
                    }
                    Kind::Spine => {
                        self.ctx.nb.spines.set_word(i, 0);
                        self.ctx.nb.spines.free(i);
                    }
                    Kind::Env => {
                        self.ctx.rp.envs.set_word(i, 0);
                        self.ctx.rp.envs.free(i);
                    }
                }
                for &(ck, cid) in &children {
                    let Some(stamp) = drop_child(c, ck, cid) else { continue };
                    if stale(stamp) {
                        work.push((ck, cid));
                    }
                }
            }
        }
        // the next sweep waits for as many allocations as half the slots
        // still present, so that scanning stays in proportion to allocating
        let present: usize = self.ctx.nb.vals.present_ranges().map(|r| r.len()).sum::<usize>()
            + self.ctx.nb.spines.present_ranges().map(|r| r.len()).sum::<usize>()
            + self.ctx.rp.envs.present_ranges().map(|r| r.len()).sum::<usize>();
        c.next_sweep = SWEEP_MIN.max(present as u64 / 2);
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
        fn live<T>(a: &Arena<T>, i: usize) -> bool { a.present(i) && a.word(i) & ALLOC != 0 }
        let mut ev = vec![0u64; self.nb.vals.len()];
        let mut es = vec![0u64; self.nb.spines.len()];
        let mut ee = vec![0u64; self.rp.envs.len()];
        es[0] += 1;
        ee[0] += 1;
        for i in (0..self.nb.vals.len()).filter(|&i| live(&self.nb.vals, i)) {
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
        for i in (1..self.nb.spines.len()).filter(|&i| live(&self.nb.spines, i)) {
            let n = &self.nb.spines[i];
            es[n.parent as usize] += 1;
            if let Elim::App(a) = n.elim {
                ev[a as usize] += 1;
            }
        }
        for i in (1..self.rp.envs.len()).filter(|&i| live(&self.rp.envs, i)) {
            let n = &self.rp.envs[i];
            ee[n.parent as usize] += 1;
            match n.entry {
                Entry::V(v) => ev[v as usize] += 1,
                Entry::Val(_, env) if env & VIEW_BIT == 0 => ee[env as usize] += 1,
                _ => {}
            }
        }
        let nb = &self.nb;
        for (&k, &r) in &nb.iota_cache {
            if let Some(r) = r.filter(|&r| r != k && live(&self.nb.vals, k as usize)) {
                ev[r as usize] += 1;
            }
        }
        for (&k, &t) in &nb.type_cache {
            if t != k && live(&self.nb.vals, k as usize) {
                ev[t as usize] += 1;
            }
        }
        for v in nb.clo_val_cache.values().chain(nb.const_val_cache.values()).chain(nb.const_ty_cache.values())
            .chain(nb.rec_rule_cache.values()).chain(nb.local_cache.values()).chain(nb.unfold_cache.values().flatten())
        {
            ev[v.id() as usize] += 1;
        }
        let mut surplus = 0u64;
        let words: [(&str, &dyn Fn(usize) -> Option<u32>, &Vec<u64>); 3] = [
            ("value", &|i| live(&self.nb.vals, i).then(|| self.nb.vals.word(i)), &ev),
            ("spine", &|i| live(&self.nb.spines, i).then(|| self.nb.spines.word(i)), &es),
            ("env", &|i| live(&self.rp.envs, i).then(|| self.rp.envs.word(i)), &ee),
        ];
        for (what, stored, expected) in words {
            for (i, &e) in expected.iter().enumerate() {
                let Some(w) = stored(i) else {
                    assert!(e == 0, "{what} {i}: freed with {e} live holders");
                    continue;
                };
                let s = u64::from(w & COUNT_MAX);
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
