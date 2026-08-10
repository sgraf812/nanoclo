//! Delayed instantiation over nanoda's arena.
//!
//! Terms in flight are closures `(ExprPtr, EnvId)`; entering a binder extends
//! an interned environment in O(1); whnf and def-eq run on spined closures;
//! substitution is forced (`reify`) only at type boundaries.
//!
//! Free variables are
//! nanoda `Local` expressions carrying their binder type, so no separate local
//! context is needed. State lives in [`CloState`], one per `TypeChecker`
//! (nanoda creates a `TypeChecker` and expression dag per declaration, so all
//! of this is per-declaration state).

use crate::expr::{BinderStyle, Expr};
use crate::tc::TypeChecker;
use crate::util::{
    new_fx_hash_map, new_fx_hash_set, ExprPtr, FxHashMap, FxHashSet,
};
use Expr::*;

pub(crate) type EnvId = u32;
pub(crate) const ENV_NIL: EnvId = 0;


#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Entry<'t> {
    Val(ExprPtr<'t>, EnvId),
    /// A neutral entry: the `Local` expression standing for the variable.
    Neu(ExprPtr<'t>),
    /// An evaluated entry, named by its value index. Pushed by evaluation
    /// when beta extends an environment with a value already in hand.
    V(crate::nbe::ValId),
}

pub(crate) struct EnvNode<'t> {
    pub(crate) entry: Entry<'t>,
    pub(crate) parent: EnvId,
    pub(crate) len: u32,
    /// one past the highest de Bruijn level bound anywhere in this chain
    pub(crate) next_level: u32,
    /// Myers jump pointer for O(log n) indexing
    pub(crate) jump: EnvId,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Clo<'t> {
    pub e: ExprPtr<'t>,
    pub env: EnvId,
}

impl<'t> Clo<'t> {
    pub(crate) fn of(e: ExprPtr<'t>) -> Clo<'t> { Clo { e, env: ENV_NIL } }
}

/// Spines are short in practice, so they live inline until they are not.
pub(crate) type SpineVec<'t> = smallvec::SmallVec<[Clo<'t>; 8]>;

#[derive(Clone)]
pub(crate) struct SClo<'t> {
    pub head: Clo<'t>,
    pub spine: SpineVec<'t>,
}

/// Injective packing of an environment-extension request into two words
/// (pointer raw bits are 32-bit).
#[inline]
fn pack_entry_key(env: EnvId, entry: Entry) -> (u64, u64) {
    match entry {
        Entry::Val(e, venv) => ((env as u64) << 32 | e.get_hash(), (venv as u64) << 1 | 1),
        Entry::Neu(e) => ((env as u64) << 32 | e.get_hash(), 0),
        Entry::V(v) => ((env as u64) << 32 | v as u64, 2),
    }
}

/// The loose bvar indices an expression reads. `Dense` is every index below
/// its range, so projecting an environment onto it changes nothing; `Mask` is
/// exact for a range that fits a word; `Wide` is a range beyond a word that is
/// not dense.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Uses {
    Dense,
    Mask(u64),
    Wide,
}

pub(crate) struct CloState<'t> {
    pub(crate) envs: Vec<EnvNode<'t>>,
    /// keyed by `pack_entry_key(parent, entry)`
    pub(crate) env_intern: FxHashMap<(u64, u64), EnvId>,

    pub(crate) infer_cache_check: FxHashMap<Clo<'t>, ExprPtr<'t>>,
    pub(crate) infer_cache_only: FxHashMap<Clo<'t>, ExprPtr<'t>>,
    pub(crate) reify_go_cache: Gen2<(ExprPtr<'t>, EnvId, u16), ExprPtr<'t>>,
    /// `e -> one past the highest de Bruijn level of an fvar occurring in it`
    pub(crate) lvl_cache: FxHashMap<ExprPtr<'t>, u32>,
    /// `type expr -> whether it is a proposition`; proof irrelevance asks
    /// this of the same few types once per comparison.
    pub(crate) prop_cache: FxHashMap<ExprPtr<'t>, bool>,
    /// `e -> the loose bvar indices it reads`
    pub(crate) umask_cache: FxHashMap<ExprPtr<'t>, Uses>,
    /// `(read set, env) -> that environment projected onto that set`. The
    /// projection depends on the set and not on the expression that induced
    /// it, so expressions reading the same positions share one projection.
    pub(crate) proj_cache: FxHashMap<(u64, EnvId), EnvId>,
    /// keyed by the packed `(ae|aenv, be|benv, aoff|boff)` triple
    pub(crate) eq_mod_cache: Gen2<(u64, u64, u32), bool>,

    // Caches valid across declarations (per thread): keys and stored values
    // are closed (fvar-free, env-free) and refer only to constants visible at
    // the time of entry, so no per-declaration state can leak through them.
    // Never populated while a temporary environment extension (nested
    // inductive checking) is active.
    /// const expr -> its level-instantiated definition value
    pub(crate) g_unfold: FxHashMap<ExprPtr<'t>, ExprPtr<'t>>,
    /// const expr -> its level-instantiated type
    pub(crate) g_inst_ty: FxHashMap<ExprPtr<'t>, ExprPtr<'t>>,

    /// diagnostic counters: [infer, whnf_core, whnf, def_eq, whnf_hit,
    /// whnf_miss, whnf_core_hit, whnf_core_miss, unfold_hit, unfold_miss,
    /// push_entry, eq_mod]
    pub(crate) ctrs: [u64; 25],
}

/// Totals over all declarations, printed at exit when `NANOCLO_CTRS` is set.
pub static G_CTRS: [std::sync::atomic::AtomicU64; 25] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 25];

pub fn ctrs_report() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    const NAMES: [&str; 25] = [
        "infer", "whnf_core", "whnf", "def_eq", "whnf_hit", "whnf_miss",
        "deq_hit", "deq_miss", "unfold_hit", "unfold_miss",
        "push_entry", "eq_mod", "eqm_hit", "eqm_miss", "eqm_fast", "eqm_nomemo",
        "inf_hit", "inf_miss", "inf_var_val", "inf_var_neu",
        "inf_local", "inf_sort", "inf_const",
        "spec_fail", "spec_abort",
    ];
    let mut out = String::new();
    for (n, c) in NAMES.iter().zip(G_CTRS.iter()) {
        out.push_str(&format!("{}={} ", n, c.load(Relaxed)));
    }
    out
}

    /// Ceiling on a single memo table within one declaration. The tables are
/// cleared between declarations, but a large declaration can grow one
/// without bound; peak memory is set by the largest declaration in the
/// export. Dropping a memo costs recomputation, never correctness.


impl<'t> CloState<'t> {
    pub(crate) fn new() -> Self {
        CloState {
            envs: vec![EnvNode { entry: Entry::Val(crate::util::Ptr::from(crate::util::DagMarker::ExportFile, 0), 0), parent: 0, len: 0, next_level: 0, jump: 0 }],
            env_intern: new_fx_hash_map(),
            infer_cache_check: new_fx_hash_map(),
            infer_cache_only: new_fx_hash_map(),
            reify_go_cache: Gen2::new(),
            lvl_cache: new_fx_hash_map(),
            prop_cache: new_fx_hash_map(),
            umask_cache: new_fx_hash_map(),
            proj_cache: new_fx_hash_map(),
            eq_mod_cache: Gen2::new(),
            g_unfold: new_fx_hash_map(),
            g_inst_ty: new_fx_hash_map(),
            ctrs: [0; 25],
        }
    }

    /// Accumulate this context's counters into the process-wide totals.
    pub(crate) fn flush_ctrs(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        for (g, c) in G_CTRS.iter().zip(self.ctrs.iter_mut()) {
            g.fetch_add(*c, Relaxed);
            *c = 0;
        }
    }

    /// Clear per-declaration state, keeping allocated capacity. The global
    /// caches persist; they only hold fvar-free, env-free entries.
    pub(crate) fn reset_decl(&mut self) {
        const CAP: usize = 1 << 14;
        fn rm<K: std::hash::Hash + Eq, V>(m: &mut FxHashMap<K, V>) {
            if m.capacity() > CAP {
                *m = FxHashMap::with_capacity_and_hasher(CAP / 2, Default::default());
            } else if !m.is_empty() {
                m.clear();
            }
        }
        fn rs<K: std::hash::Hash + Eq>(m: &mut FxHashSet<K>) {
            if m.capacity() > CAP {
                *m = FxHashSet::with_capacity_and_hasher(CAP / 2, Default::default());
            } else if !m.is_empty() {
                m.clear();
            }
        }
        self.envs.truncate(1);
        rm(&mut self.env_intern);
        rm(&mut self.infer_cache_check);
        rm(&mut self.infer_cache_only);
        self.reify_go_cache.reset_decl();
        rm(&mut self.lvl_cache);
        rm(&mut self.prop_cache);
        rm(&mut self.umask_cache);
        rm(&mut self.proj_cache);
        self.eq_mod_cache.reset_decl();
    }
}


/// Entry ceiling per memo table.
const CCAP: usize = 1 << 20;

/// Two-generation memo: entries land in `new`; when it fills, `new` becomes
/// `old` and a fresh `new` starts. A hit in `old` is promoted, so entries
/// that keep being used survive the swaps.
pub(crate) struct Gen2<K, V> {
    pub(crate) new: FxHashMap<K, V>,
    pub(crate) old: FxHashMap<K, V>,
}

impl<K: std::hash::Hash + Eq + Copy, V: Copy> Gen2<K, V> {
    pub(crate) fn new() -> Self {
        Gen2 { new: new_fx_hash_map(), old: new_fx_hash_map() }
    }

    #[inline]
    pub(crate) fn get(&mut self, k: &K) -> Option<V> {
        if let Some(v) = self.new.get(k) {
            return Some(*v);
        }
        let v = self.old.get(k).copied()?;
        self.insert(*k, v);
        Some(v)
    }

    #[inline]
    pub(crate) fn insert(&mut self, k: K, v: V) {
        if self.new.len() >= CCAP {
            std::mem::swap(&mut self.new, &mut self.old);
            self.new.clear();
        }
        self.new.insert(k, v);
    }

    pub(crate) fn reset_decl(&mut self) {
        const KEEP: usize = 1 << 14;
        for m in [&mut self.new, &mut self.old] {
            if m.capacity() > KEEP {
                *m = new_fx_hash_map();
            } else {
                m.clear();
            }
        }
    }
}

impl<'x, 't: 'x, 'p: 't> TypeChecker<'x, 't, 'p> {
    #[inline]
    pub(crate) fn lbr(&self, e: ExprPtr<'t>) -> u16 { self.ctx.num_loose_bvars(e) }

    // ---- environments ----

    pub(crate) fn env_len(&self, env: EnvId) -> u32 { self.ctx.rp.envs[env as usize].len }

    pub(crate) fn env_next_level(&self, env: EnvId) -> u32 { self.ctx.rp.envs[env as usize].next_level }

    /// One past the highest de Bruijn level carried by a free variable in
    /// `e`. Free variables reach an expression only from the context it was
    /// built in, so this is bounded by the depth of that context.
    pub(crate) fn max_level(&mut self, e: ExprPtr<'t>) -> u32 {
        if !self.ctx.has_fvars(e) {
            return 0;
        }
        if let Some(&r) = self.ctx.rp.lvl_cache.get(&e) {
            return r;
        }
        let r = match self.ctx.read_expr(e) {
            Local { id: crate::expr::FVarId::DbjLevel(l), binder_type, .. } => {
                self.max_level(binder_type).max(l + 1)
            }
            Local { binder_type, .. } => self.max_level(binder_type),
            App { fun, arg, .. } => self.max_level(fun).max(self.max_level(arg)),
            Lambda { binder_type, body, .. } | Pi { binder_type, body, .. } => {
                self.max_level(binder_type).max(self.max_level(body))
            }
            Let { binder_type, val, body, .. } => self
                .max_level(binder_type)
                .max(self.max_level(val))
                .max(self.max_level(body)),
            Proj { structure, .. } => self.max_level(structure),
            _ => 0,
        };
        self.ctx.rp.lvl_cache.insert(e, r);
        r
    }

    /// The de Bruijn level to give a binder opened inside the closure
    /// `(e, env)`: one past every level reachable from it, so the variable
    /// that names that binder cannot be confused with one already in scope.
    pub(crate) fn next_level(&mut self, e: ExprPtr<'t>, env: EnvId) -> u32 {
        self.max_level(e).max(self.env_next_level(env))
    }

    /// Extend an environment with an evaluated entry. Everything the new
    /// node holds comes from the parent node, so a miss costs one probe.
    pub(crate) fn push_entry_v(&mut self, env: EnvId, v: crate::nbe::ValId) -> EnvId {
        self.ctx.rp.ctrs[10] += 1;
        let key = pack_entry_key(env, Entry::V(v));
        let crate::closure::CloState { envs, env_intern, .. } = &mut self.ctx.rp;
        match env_intern.entry(key) {
            std::collections::hash_map::Entry::Occupied(o) => *o.get(),
            std::collections::hash_map::Entry::Vacant(slot) => {
                let p = &envs[env as usize];
                let len = p.len + 1;
                let next_level = p.next_level;
                let jump = {
                    let d1 = p.len - envs[p.jump as usize].len;
                    let j = &envs[p.jump as usize];
                    let d2 = j.len - envs[j.jump as usize].len;
                    if d1 == d2 { j.jump } else { env }
                };
                let id = u32::try_from(envs.len()).unwrap();
                envs.push(EnvNode { entry: Entry::V(v), parent: env, len, next_level, jump });
                slot.insert(id);
                id
            }
        }
    }

    pub(crate) fn push_entry(&mut self, env: EnvId, entry: Entry<'t>) -> EnvId {
        self.ctx.rp.ctrs[10] += 1;
        // The entry is part of the identity of every environment built on
        // top of it, so it enters in normal form: a value that reads nothing
        // from its own environment carries none, and a value that is a
        // variable is replaced by what that variable stands for. Without
        // this, one binding reached along two paths yields two environments
        // and splits every cache keyed on them.
        let entry = match entry {
            Entry::Val(e, venv) if venv != ENV_NIL => {
                match self.norm_clo(Clo { e, env: venv }) {
                    Clo { e, env } => Entry::Val(e, env),
                }
            }
            e => e,
        };
        let key = pack_entry_key(env, entry);
        if let Some(&id) = self.ctx.rp.env_intern.get(&key) {
            return id;
        }
        let len = self.env_len(env) + 1;
        let next_level = {
            let parent = self.env_next_level(env);
            match entry {
                Entry::Neu(fv) => parent.max(self.max_level(fv)),
                Entry::Val(e, venv) => {
                    parent.max(self.max_level(e)).max(self.env_next_level(venv))
                }
                // An evaluated entry introduces no Local the environment has
                // not seen: every neutral inside it entered through an
                // enclosing environment, whose level this already covers.
                Entry::V(_) => parent,
            }
        };
        // Myers jump: if dist(parent) == dist(parent.jump), jump to parent.jump.jump
        let p = &self.ctx.rp.envs[env as usize];
        let jump = {
            let d1 = p.len - self.ctx.rp.envs[p.jump as usize].len;
            let j = &self.ctx.rp.envs[p.jump as usize];
            let d2 = j.len - self.ctx.rp.envs[j.jump as usize].len;
            if d1 == d2 { j.jump } else { env }
        };
        let id = u32::try_from(self.ctx.rp.envs.len()).unwrap();
        self.ctx.rp.envs.push(EnvNode { entry, parent: env, len, next_level, jump });
        self.ctx.rp.env_intern.insert(key, id);
        id
    }

    /// Which loose bvar indices `e` reads. Union at an application is a
    /// bitwise or and passing a binder is a shift, both constant time, and a
    /// set covering everything below the range needs no representation.
    pub(crate) fn uses_mask(&mut self, e: ExprPtr<'t>) -> Uses {
        let n = self.ctx.read_expr(e);
        let lbr = n.num_loose_bvars();
        if lbr == 0 {
            return Uses::Mask(0);
        }
        if let Some(&u) = self.ctx.rp.umask_cache.get(&e) {
            return u;
        }
        fn join(a: Uses, b: Uses) -> Uses {
            match (a, b) {
                (Uses::Mask(x), Uses::Mask(y)) => Uses::Mask(x | y),
                _ => Uses::Wide,
            }
        }
        fn under(u: Uses) -> Uses {
            match u {
                Uses::Mask(m) => Uses::Mask(m >> 1),
                other => other,
            }
        }
        let u = match n {
            Var { dbj_idx, .. } if dbj_idx < 64 => Uses::Mask(1u64 << dbj_idx),
            Var { .. } => Uses::Wide,
            App { fun, arg, .. } => {
                let a = self.uses_mask(fun);
                let b = self.uses_mask(arg);
                join(a, b)
            }
            Lambda { binder_type, body, .. } | Pi { binder_type, body, .. } => {
                let d = self.uses_mask(binder_type);
                let b = self.uses_mask(body);
                join(d, under(b))
            }
            Let { binder_type, val, body, .. } => {
                let t = self.uses_mask(binder_type);
                let v = self.uses_mask(val);
                let b = self.uses_mask(body);
                join(join(t, v), under(b))
            }
            Proj { structure, .. } => self.uses_mask(structure),
            _ => Uses::Mask(0),
        };
        let u = match u {
            Uses::Mask(m) if lbr <= 64 && m == u64::MAX >> (64 - lbr) => Uses::Dense,
            u => u,
        };
        self.ctx.rp.umask_cache.insert(e, u);
        u
    }

    /// entry for de Bruijn index `i` (0 = innermost)
    pub(crate) fn lookup(&self, env: EnvId, i: u16) -> Entry<'t> {
        let len = self.env_len(env);
        assert!(
            (i as u32) < len,
            "loose bvar: index {} in environment of length {} (env id {})",
            i,
            len,
            env
        );
        let mut cur = env;
        let target = len - 1 - i as u32; // 0-based from bottom; node with len == target+1
        loop {
            let n = &self.ctx.rp.envs[cur as usize];
            debug_assert!(n.len > target);
            if n.len == target + 1 {
                return n.entry;
            }
            let j = &self.ctx.rp.envs[n.jump as usize];
            if j.len > target {
                cur = n.jump;
            } else {
                cur = n.parent;
            }
        }
    }

    /// The free variable standing for a binder opened over an environment of
    /// `level` entries. Naming it by that level, rather than by a counter,
    /// makes two openings of the same binder produce the same variable, so
    /// the closures, environments and cache keys built on top of it coincide
    /// instead of being fresh every time. The binder type is part of the
    /// variable's identity, so a level shared by two binders of different
    /// types still gives two variables.
    pub(crate) fn fvar_at(&mut self, level: u32, ty: ExprPtr<'t>) -> ExprPtr<'t> {
        let anon = self.ctx.anonymous();
        self.ctx.remake_dbj_level(anon, BinderStyle::Default, ty, level)
    }

    pub(crate) fn fvar_type(&self, fv: ExprPtr<'t>) -> ExprPtr<'t> {
        match self.ctx.read_expr(fv) {
            Local { binder_type, .. } => binder_type,
            _ => unreachable!("fvar_type on non-Local"),
        }
    }

    /// The identity of a closure as a question: the expression together with
    /// the bindings that expression reads. Everything else in the environment
    /// is invisible to it, so closures differing only there ask the same
    /// question and share one answer.
    pub(crate) fn key(&mut self, c: Clo<'t>) -> Clo<'t> {
        if c.env == ENV_NIL {
            return c;
        }
        let mask = match self.uses_mask(c.e) {
            Uses::Mask(0) => return Clo::of(c.e),
            Uses::Mask(m) => m,
            Uses::Dense | Uses::Wide => return c,
        };
        Clo { e: c.e, env: self.project(mask, c.env) }
    }
    /// The environment cut down to the entries a read set names, rebuilt so
    /// that environments agreeing on those entries become the same one.
    /// Yields `env` untouched when the set names an entry the environment
    /// does not have: that is a coarser key rather than a wrong one, and a
    /// term whose variables really do escape is caught by `lookup`.
    fn project(&mut self, mask: u64, env: EnvId) -> EnvId {
        if mask == 0 {
            return ENV_NIL;
        }
        if let Some(&p) = self.ctx.rp.proj_cache.get(&(mask, env)) {
            return p;
        }
        // Every index in the mask is below 64, so the entries it names sit
        // within the first 64 links: one walk collects them all.
        let mut picked: [Option<Entry<'t>>; 64] = [None; 64];
        let mut cur = env;
        for d in 0..64 {
            if cur == ENV_NIL {
                break;
            }
            let node = &self.ctx.rp.envs[cur as usize];
            if mask & (1u64 << d) != 0 {
                picked[d] = Some(node.entry);
            }
            if mask >> d == 1 {
                break;
            }
            cur = node.parent;
        }
        let mut proj = ENV_NIL;
        let mut m = mask;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            m &= m - 1;
            let Some(entry) = picked[i] else { return env };
            proj = self.push_entry(proj, entry);
        }
        self.ctx.rp.proj_cache.insert((mask, env), proj);
        proj
    }


    /// Cache-key normalization: resolve bvar heads to the entry they denote;
    /// a closed result drops its environment.
    pub(crate) fn norm_clo(&mut self, c: Clo<'t>) -> Clo<'t> {
        let n = self.ctx.read_expr(c.e);
        let (c, lbr) = if matches!(n, Var { .. }) {
            let (e2, env2) = self.chase(c.e, c.env);
            (Clo { e: e2, env: env2 }, self.lbr(e2))
        } else {
            (c, n.num_loose_bvars())
        };
        if lbr == 0 {
            return Clo::of(c.e);
        }
        c
    }

    // ---- reify ----

    pub(crate) fn reify(&mut self, c: Clo<'t>) -> ExprPtr<'t> {
        if c.env == ENV_NIL || self.lbr(c.e) == 0 {
            return c.e;
        }
        self.reify_go(c.env, 0, c.e)
    }

    pub(crate) fn reify_go(&mut self, env: EnvId, offset: u16, e: ExprPtr<'t>) -> ExprPtr<'t> {
        let n = self.ctx.read_expr(e);
        if n.num_loose_bvars() <= offset {
            return e;
        }
        let memo_key = (e, env, offset);
        if let Some(r) = self.ctx.rp.reify_go_cache.get(&memo_key) {
            return r;
        }
        let r = match n {
            Var { dbj_idx, .. } => match self.lookup(env, dbj_idx - offset) {
                Entry::V(v) => self.nb_readback(v),
                Entry::Neu(fv) => fv,
                Entry::Val(e2, env2) => {
                    if env2 == ENV_NIL {
                        e2
                    } else {
                        self.reify(Clo { e: e2, env: env2 })
                    }
                }
            },
            App { fun, arg, .. } => {
                let f2 = self.reify_go(env, offset, fun);
                let x2 = self.reify_go(env, offset, arg);
                self.ctx.mk_app(f2, x2)
            }
            Lambda { binder_name, binder_style, binder_type, body, .. } => {
                let d2 = self.reify_go(env, offset, binder_type);
                let b2 = self.reify_go(env, offset + 1, body);
                self.ctx.mk_lambda(binder_name, binder_style, d2, b2)
            }
            Pi { binder_name, binder_style, binder_type, body, .. } => {
                let d2 = self.reify_go(env, offset, binder_type);
                let b2 = self.reify_go(env, offset + 1, body);
                self.ctx.mk_pi(binder_name, binder_style, d2, b2)
            }
            Let { binder_name, binder_type, val, body, nondep, .. } => {
                let t2 = self.reify_go(env, offset, binder_type);
                let v2 = self.reify_go(env, offset, val);
                let b2 = self.reify_go(env, offset + 1, body);
                self.ctx.mk_let(binder_name, t2, v2, b2, nondep)
            }
            Proj { ty_name, idx, structure, .. } => {
                let x2 = self.reify_go(env, offset, structure);
                self.ctx.mk_proj(ty_name, idx, x2)
            }
            _ => e,
        };
        self.ctx.rp.reify_go_cache.insert(memo_key, r);
        r
    }

    // ---- eqMod: structural equality modulo substitution ----

    pub(crate) fn eq_mod(
        &mut self,
        ae: ExprPtr<'t>,
        aenv: EnvId,
        aoff: u16,
        be: ExprPtr<'t>,
        benv: EnvId,
        boff: u16,
    ) -> bool {
        self.ctx.rp.ctrs[11] += 1;
        // resolve a-side head indirections
        let an = self.ctx.read_expr(ae);
        if let Var { dbj_idx: i, .. } = an {
            if i >= aoff {
                return match self.lookup(aenv, i - aoff) {
                    Entry::Val(e2, env2) => self.eq_mod(e2, env2, 0, be, benv, boff),
                    Entry::Neu(fv) => self.eq_mod_neu(fv, be, benv, boff),
                    Entry::V(_) => return false,
                };
            }
        }
        // resolve b-side
        let bn = self.ctx.read_expr(be);
        if let Var { dbj_idx: j, .. } = bn {
            if j >= boff {
                return match self.lookup(benv, j - boff) {
                    Entry::Val(e2, env2) => self.eq_mod(ae, aenv, aoff, e2, env2, 0),
                    Entry::Neu(fv) => self.eq_mod_neu(fv, ae, aenv, aoff),
                    Entry::V(_) => return false,
                };
            }
        }
        // fast paths
        let albr = an.num_loose_bvars();
        if albr <= aoff && bn.num_loose_bvars() <= boff && aoff == boff {
            self.ctx.rp.ctrs[14] += 1;
            return ae == be;
        }
        if ae == be && aenv == benv && aoff == boff {
            self.ctx.rp.ctrs[14] += 1;
            return true;
        }
        if ae == be && albr <= aoff.min(boff) {
            self.ctx.rp.ctrs[14] += 1;
            return true;
        }
        // Consulted from either call order, so the test must not depend on
        // which side is which: the key is canonical in the two sides.
        let is_composite = |n: &Expr<'t>| {
            matches!(n, App { .. } | Lambda { .. } | Pi { .. } | Let { .. } | Proj { .. })
        };
        let composite = is_composite(&an) || is_composite(&bn);
        // The key is the six arguments packed losslessly: an environment id
        // and an offset are only part of the question when the expression
        // still has loose bvars at that offset, so a side that is closed
        // there keys as (e, NIL, 0). eq_mod is symmetric, so the two sides
        // are ordered canonically and each question is stored once.
        let blbr = bn.num_loose_bvars();
        let (ka_env, ka_off) = if albr <= aoff { (ENV_NIL, 0u16) } else { (aenv, aoff) };
        let (kb_env, kb_off) = if blbr <= boff { (ENV_NIL, 0u16) } else { (benv, boff) };
        let ka = (ka_env as u64) << 32 | ae.get_hash();
        let kb = (kb_env as u64) << 32 | be.get_hash();
        let memo_key = if ka <= kb {
            (ka, kb, (ka_off as u32) << 16 | kb_off as u32)
        } else {
            (kb, ka, (kb_off as u32) << 16 | ka_off as u32)
        };
        if composite {
            if let Some(r) = self.ctx.rp.eq_mod_cache.get(&memo_key) {
                self.ctx.rp.ctrs[12] += 1;
                return r;
            }
            self.ctx.rp.ctrs[13] += 1;
        } else {
            self.ctx.rp.ctrs[15] += 1;
        }
        let r = match (an, bn) {
            (Var { dbj_idx: i, .. }, Var { dbj_idx: j, .. }) => i == j,
            (Local { .. }, Local { .. }) => ae == be,
            (Const { name: c1, levels: l1, .. }, Const { name: c2, levels: l2, .. }) => {
                c1 == c2 && l1 == l2
            }
            (Sort { level: l1, .. }, Sort { level: l2, .. }) => l1 == l2,
            (NatLit { ptr: v1, .. }, NatLit { ptr: v2, .. }) => v1 == v2,
            (StringLit { ptr: s1, .. }, StringLit { ptr: s2, .. }) => s1 == s2,
            (App { fun: f1, arg: a1, .. }, App { fun: f2, arg: a2, .. }) => {
                self.eq_mod(f1, aenv, aoff, f2, benv, boff)
                    && self.eq_mod(a1, aenv, aoff, a2, benv, boff)
            }
            (
                Lambda { binder_type: d1, body: b1, .. },
                Lambda { binder_type: d2, body: b2, .. },
            )
            | (Pi { binder_type: d1, body: b1, .. }, Pi { binder_type: d2, body: b2, .. }) => {
                self.eq_mod(d1, aenv, aoff, d2, benv, boff)
                    && self.eq_mod(b1, aenv, aoff + 1, b2, benv, boff + 1)
            }
            (
                Let { binder_type: t1, val: v1, body: b1, .. },
                Let { binder_type: t2, val: v2, body: b2, .. },
            ) => {
                self.eq_mod(t1, aenv, aoff, t2, benv, boff)
                    && self.eq_mod(v1, aenv, aoff, v2, benv, boff)
                    && self.eq_mod(b1, aenv, aoff + 1, b2, benv, boff + 1)
            }
            (
                Proj { ty_name: n1, idx: i1, structure: e1, .. },
                Proj { ty_name: n2, idx: i2, structure: e2, .. },
            ) => n1 == n2 && i1 == i2 && self.eq_mod(e1, aenv, aoff, e2, benv, boff),
            _ => false,
        };
        if composite {
            self.ctx.rp.eq_mod_cache.insert(memo_key, r);
        }
        r
    }

    /// Does the closure `(e, env, off)` denote exactly the free variable `fv`?
    pub(crate) fn eq_mod_neu(&mut self, fv: ExprPtr<'t>, e: ExprPtr<'t>, env: EnvId, off: u16) -> bool {
        if let Var { dbj_idx: j, .. } = self.ctx.read_expr(e) {
            if j >= off {
                return match self.lookup(env, j - off) {
                    Entry::Val(e2, env2) => self.eq_mod_neu(fv, e2, env2, 0),
                    Entry::Neu(g) => fv == g,
                    Entry::V(_) => return false,
                };
            }
            return false;
        }
        e == fv
    }

    /// Follow bvar -> Val chains to their base. Neu entries resolve to the
    /// fvar expr. Returns the input unchanged only for non-bvar heads.
    pub(crate) fn chase(&mut self, e0: ExprPtr<'t>, env0: EnvId) -> (ExprPtr<'t>, EnvId) {
        let mut e = e0;
        let mut env = env0;
        loop {
            let i = match self.ctx.read_expr(e) {
                Var { dbj_idx, .. } => dbj_idx,
                _ => break,
            };
            match self.lookup(env, i) {
                Entry::Neu(fv) => {
                    e = fv;
                    env = ENV_NIL;
                    break;
                }
                Entry::V(_) => break,
                Entry::Val(e2, env2) => {
                    e = e2;
                    env = env2;
                }
            }
        }
        (e, env)
    }
}
