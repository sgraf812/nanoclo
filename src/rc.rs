//! Reference counts for the evaluator's values, spines and environments.
//!
//! A node's count is the number of its holders: the nodes that name it as a
//! child, the forced cell or domain that records it, the caches that keep it,
//! and the handles (`V`, `S`, `E`) held by the evaluator. Nodes and weak
//! tables name each other by raw id; a handle is a raw id that holds a
//! reference, taken by `own` and given back when it drops.
//!
//! The counts live apart from the nodes, in a store that only this module
//! touches, so a handle can update its count while the evaluator holds the
//! arenas mutably. Each checker context installs its own store for as long as
//! it lives; a context created while another one is alive stacks on top.

use crate::arena::Arena;
use std::cell::{Cell, RefCell};

pub(crate) struct Counts {
    pub(crate) vals: Arena<u32>,
    pub(crate) spines: Arena<u32>,
    pub(crate) envs: Arena<u32>,
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
        let mut c = Box::new(Counts { vals: Arena::new(), spines: Arena::new(), envs: Arena::new() });
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
    // and only this module and the arena push paths reach it, one access at
    // a time on the owning thread.
    unsafe { &mut *CURRENT.with(|cur| cur.get()) }
}

macro_rules! handle {
    ($name:ident, $field:ident, $doc:literal) => {
        #[doc = $doc]
        pub(crate) struct $name(u32);

        impl $name {
            /// Take a reference to node `id`.
            #[inline]
            pub(crate) fn own(id: u32) -> Self {
                inc(&mut counts().$field, id);
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
            fn drop(&mut self) { dec(&mut counts().$field, self.0) }
        }

        impl PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
        }
        impl Eq for $name {}
    };
}

handle!(V, vals, "A value held by the evaluator.");
handle!(S, spines, "A spine held by the evaluator.");
handle!(E, envs, "An environment held by the evaluator.");

#[inline]
pub(crate) fn inc(a: &mut Arena<u32>, id: u32) { a[id as usize] += 1; }

#[inline]
pub(crate) fn dec(a: &mut Arena<u32>, id: u32) {
    let c = &mut a[id as usize];
    *c = c.checked_sub(1).expect("reference count below zero");
}

/// Take a reference to value `id`.
#[inline]
pub(crate) fn inc_val(id: u32) { inc(&mut counts().vals, id) }
/// Take a reference to spine `id`.
#[inline]
pub(crate) fn inc_spine(id: u32) { inc(&mut counts().spines, id) }
/// Take a reference to environment `id`.
#[inline]
pub(crate) fn inc_env(id: u32) { inc(&mut counts().envs, id) }
/// Pin environment `id` for the rest of the declaration.
#[inline]
pub(crate) fn pin_env(id: u32) {
    inc_env(id);
    PINS.with(|p| p.set(p.get() + 1));
}
/// Give back a reference to value `id`.
#[inline]
pub(crate) fn dec_val(id: u32) { dec(&mut counts().vals, id) }

impl<'t, 'p> crate::util::TcCtx<'t, 'p> {
    /// Recompute every count from the node graph and the caches that hold
    /// references, and panic unless each stored count covers its expected
    /// one and the surplus is exactly the pins. Run when no handle is alive.
    pub(crate) fn check_counts(&self) {
        use crate::closure::{Entry, VIEW_BIT};
        use crate::nbe::{Elim, RigidHead, Value};
        let c = counts();
        let mut ev = vec![0u64; self.nb.vals.len()];
        let mut es = vec![0u64; self.nb.spines.len()];
        let mut ee = vec![0u64; self.rp.envs.len()];
        es[0] += 1;
        ee[0] += 1;
        for i in 0..self.nb.vals.len() {
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
        for i in 1..self.nb.spines.len() {
            let n = &self.nb.spines[i];
            es[n.parent as usize] += 1;
            if let Elim::App(a) = n.elim {
                ev[a as usize] += 1;
            }
        }
        for i in 1..self.rp.envs.len() {
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
                let s = u64::from(stored[i]);
                assert!(s >= e, "{what} {i}: count {s} below its {e} holders");
                surplus += s - e;
            }
        }
        let pins = PINS.with(|p| p.get());
        assert!(surplus == pins, "counts exceed their holders by {surplus}, but {pins} references are pinned");
    }
}
