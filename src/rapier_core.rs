//! The rapier delayed-instantiation core over nanoda's arena.
//!
//! Terms in flight are closures `(ExprPtr, EnvId)`; entering a binder extends
//! an interned environment in O(1); whnf and def-eq run on spined closures;
//! substitution is forced (`reify`) only at type boundaries.
//!
//! This mirrors rapier's `checker.rs` function by function. Free variables are
//! nanoda `Local` expressions carrying their binder type, so no separate local
//! context is needed. State lives in [`RapierSt`], one per `TypeChecker`
//! (nanoda creates a `TypeChecker` and expression dag per declaration, so all
//! of this is per-declaration state).

use crate::env::{Declar, ReducibilityHint};
use crate::expr::{BinderStyle, Expr};
use crate::tc::{InferFlag, TypeChecker};
use crate::util::{
    nat_div, nat_gcd, nat_land, nat_lor, nat_mod, nat_sub, nat_xor, new_fx_hash_map,
    new_fx_hash_set, ExprPtr, FxHashMap, FxHashSet, LevelPtr, LevelsPtr, NamePtr,
};
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use std::cmp::Ordering;
use Expr::*;
use InferFlag::*;

pub(crate) type EnvId = u32;
pub(crate) const ENV_NIL: EnvId = 0;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Entry<'t> {
    Val(ExprPtr<'t>, EnvId),
    /// A neutral entry: the `Local` expression standing for the variable.
    Neu(ExprPtr<'t>),
}

pub(crate) struct EnvNode<'t> {
    entry: Entry<'t>,
    parent: EnvId,
    len: u32,
    /// Myers jump pointer for O(log n) indexing
    jump: EnvId,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Clo<'t> {
    pub e: ExprPtr<'t>,
    pub env: EnvId,
}

impl<'t> Clo<'t> {
    pub(crate) fn of(e: ExprPtr<'t>) -> Clo<'t> { Clo { e, env: ENV_NIL } }
    #[inline]
    fn okey(&self) -> (u64, u32) { (self.e.get_hash(), self.env) }
}

#[derive(Clone)]
pub(crate) struct SClo<'t> {
    pub head: Clo<'t>,
    pub spine: Vec<Clo<'t>>,
}

/// spined-closure cache key
type SKey<'t> = (Clo<'t>, Box<[Clo<'t>]>);

fn clo_le(a: &Clo, b: &Clo) -> bool { a.okey() <= b.okey() }

fn skey_le(a: &SKey, b: &SKey) -> bool {
    match a.0.okey().cmp(&b.0.okey()) {
        Ordering::Less => return true,
        Ordering::Greater => return false,
        Ordering::Equal => {}
    }
    let (x, y) = (&a.1, &b.1);
    for (c, d) in x.iter().zip(y.iter()) {
        match c.okey().cmp(&d.okey()) {
            Ordering::Less => return true,
            Ordering::Greater => return false,
            Ordering::Equal => {}
        }
    }
    x.len() <= y.len()
}

pub(crate) struct RapierSt<'t> {
    envs: Vec<EnvNode<'t>>,
    env_intern: FxHashMap<(EnvId, Entry<'t>), EnvId>,

    infer_cache_check: FxHashMap<Clo<'t>, ExprPtr<'t>>,
    infer_cache_only: FxHashMap<Clo<'t>, ExprPtr<'t>>,
    infer_s_cache: FxHashMap<SKey<'t>, ExprPtr<'t>>,
    whnf_cache: FxHashMap<Clo<'t>, SClo<'t>>,
    eq_pos: FxHashSet<(Clo<'t>, Clo<'t>)>,
    eq_neg: FxHashSet<(Clo<'t>, Clo<'t>)>,
    eq_s_pos: FxHashSet<(SKey<'t>, SKey<'t>)>,
    eq_s_neg: FxHashSet<(SKey<'t>, SKey<'t>)>,
    reify_cache: FxHashMap<Clo<'t>, ExprPtr<'t>>,
    reify_go_cache: FxHashMap<(ExprPtr<'t>, EnvId, u16), ExprPtr<'t>>,
    chase_cache: FxHashMap<(ExprPtr<'t>, EnvId), (ExprPtr<'t>, EnvId)>,
    restr_cache: FxHashMap<(u16, EnvId), EnvId>,
    abs_cache: FxHashMap<(ExprPtr<'t>, ExprPtr<'t>, u16), ExprPtr<'t>>,
    eq_mod_cache: FxHashMap<(ExprPtr<'t>, EnvId, u16, ExprPtr<'t>, EnvId, u16), bool>,
    clo_fvar_cache: FxHashMap<(ExprPtr<'t>, EnvId, u16), bool>,
    lvl_eq_cache: FxHashMap<(LevelPtr<'t>, LevelPtr<'t>), bool>,
}

impl<'t> RapierSt<'t> {
    pub(crate) fn new() -> Self {
        RapierSt {
            envs: vec![EnvNode { entry: Entry::Val(crate::util::Ptr::from(crate::util::DagMarker::ExportFile, 0), 0), parent: 0, len: 0, jump: 0 }],
            env_intern: new_fx_hash_map(),
            infer_cache_check: new_fx_hash_map(),
            infer_cache_only: new_fx_hash_map(),
            infer_s_cache: new_fx_hash_map(),
            whnf_cache: new_fx_hash_map(),
            eq_pos: new_fx_hash_set(),
            eq_neg: new_fx_hash_set(),
            eq_s_pos: new_fx_hash_set(),
            eq_s_neg: new_fx_hash_set(),
            reify_cache: new_fx_hash_map(),
            reify_go_cache: new_fx_hash_map(),
            chase_cache: new_fx_hash_map(),
            restr_cache: new_fx_hash_map(),
            abs_cache: new_fx_hash_map(),
            eq_mod_cache: new_fx_hash_map(),
            clo_fvar_cache: new_fx_hash_map(),
            lvl_eq_cache: new_fx_hash_map(),
        }
    }
}

impl<'x, 't: 'x, 'p: 't> TypeChecker<'x, 't, 'p> {
    #[inline]
    fn lbr(&self, e: ExprPtr<'t>) -> u16 { self.ctx.num_loose_bvars(e) }

    // ---- environments ----

    fn rp_env_len(&self, env: EnvId) -> u32 { self.rp.envs[env as usize].len }

    fn rp_push_entry(&mut self, env: EnvId, entry: Entry<'t>) -> EnvId {
        if let Some(&id) = self.rp.env_intern.get(&(env, entry)) {
            return id;
        }
        let len = self.rp_env_len(env) + 1;
        // Myers jump: if dist(parent) == dist(parent.jump), jump to parent.jump.jump
        let p = &self.rp.envs[env as usize];
        let jump = {
            let d1 = p.len - self.rp.envs[p.jump as usize].len;
            let j = &self.rp.envs[p.jump as usize];
            let d2 = j.len - self.rp.envs[j.jump as usize].len;
            if d1 == d2 { j.jump } else { env }
        };
        let id = u32::try_from(self.rp.envs.len()).unwrap();
        self.rp.envs.push(EnvNode { entry, parent: env, len, jump });
        self.rp.env_intern.insert((env, entry), id);
        id
    }

    /// entry for de Bruijn index `i` (0 = innermost)
    fn rp_lookup(&self, env: EnvId, i: u16) -> Entry<'t> {
        let len = self.rp_env_len(env);
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
            let n = &self.rp.envs[cur as usize];
            debug_assert!(n.len > target);
            if n.len == target + 1 {
                return n.entry;
            }
            let j = &self.rp.envs[n.jump as usize];
            if j.len > target {
                cur = n.jump;
            } else {
                cur = n.parent;
            }
        }
    }

    fn rp_fresh_fvar(&mut self, ty: ExprPtr<'t>) -> ExprPtr<'t> {
        let anon = self.ctx.anonymous();
        self.ctx.mk_unique(anon, BinderStyle::Default, ty)
    }

    fn rp_fvar_type(&self, fv: ExprPtr<'t>) -> ExprPtr<'t> {
        match self.ctx.read_expr(fv) {
            Local { binder_type, .. } => binder_type,
            _ => unreachable!("rp_fvar_type on non-Local"),
        }
    }

    /// E0 key normalization: a closed expression ignores its environment.
    #[inline]
    fn rp_key(&self, c: Clo<'t>) -> Clo<'t> {
        if c.env != ENV_NIL && self.lbr(c.e) == 0 {
            Clo::of(c.e)
        } else {
            c
        }
    }

    /// Cache-key normalization: resolve bvar heads to the entry they denote,
    /// then restrict σ to the loose-bvar range of e (entries beyond it cannot
    /// influence the result), recursively canonicalizing val entries. Two
    /// closures that denote the same substitution instance then share keys.
    fn rp_norm_clo(&mut self, c: Clo<'t>) -> Clo<'t> {
        let c = if matches!(self.ctx.read_expr(c.e), Var { .. }) {
            let (e2, env2) = self.rp_chase(c.e, c.env);
            Clo { e: e2, env: env2 }
        } else {
            c
        };
        let lbr = self.lbr(c.e);
        if lbr == 0 {
            return Clo::of(c.e);
        }
        let env = self.rp_restrict_env(lbr, c.env);
        Clo { e: c.e, env }
    }

    /// Content-canonical environment containing exactly the first `lbr`
    /// entries of `env` (all a term with loose-bvar range `lbr` can reach).
    fn rp_restrict_env(&mut self, lbr: u16, env: EnvId) -> EnvId {
        if lbr == 0 || env == ENV_NIL {
            return ENV_NIL;
        }
        let key = (lbr, env);
        if let Some(&r) = self.rp.restr_cache.get(&key) {
            return r;
        }
        let mut out = ENV_NIL;
        for i in (0..lbr).rev() {
            let ent = match self.rp_lookup(env, i) {
                Entry::Val(e2, s2) => {
                    let l2 = self.lbr(e2);
                    let r2 = self.rp_restrict_env(l2, s2);
                    Entry::Val(e2, r2)
                }
                n @ Entry::Neu(_) => n,
            };
            out = self.rp_push_entry(out, ent);
        }
        self.rp.restr_cache.insert(key, out);
        out
    }

    // ---- reify ----

    pub(crate) fn rp_reify(&mut self, c: Clo<'t>) -> ExprPtr<'t> {
        if c.env == ENV_NIL || self.lbr(c.e) == 0 {
            return c.e;
        }
        if let Some(&r) = self.rp.reify_cache.get(&c) {
            return r;
        }
        let ck = self.rp_norm_clo(c);
        if let Some(&r) = self.rp.reify_cache.get(&ck) {
            return r;
        }
        let r = self.rp_reify_go(c.env, 0, c.e);
        self.rp.reify_cache.insert(ck, r);
        r
    }

    fn rp_reify_go(&mut self, env: EnvId, offset: u16, e: ExprPtr<'t>) -> ExprPtr<'t> {
        if self.lbr(e) <= offset {
            return e;
        }
        let memo_key = (e, env, offset);
        if let Some(&r) = self.rp.reify_go_cache.get(&memo_key) {
            return r;
        }
        let r = match self.ctx.read_expr(e) {
            Var { dbj_idx, .. } => match self.rp_lookup(env, dbj_idx - offset) {
                Entry::Neu(fv) => fv,
                Entry::Val(e2, env2) => {
                    if env2 == ENV_NIL {
                        e2
                    } else {
                        self.rp_reify(Clo { e: e2, env: env2 })
                    }
                }
            },
            App { fun, arg, .. } => {
                let f2 = self.rp_reify_go(env, offset, fun);
                let x2 = self.rp_reify_go(env, offset, arg);
                self.ctx.mk_app(f2, x2)
            }
            Lambda { binder_name, binder_style, binder_type, body, .. } => {
                let d2 = self.rp_reify_go(env, offset, binder_type);
                let b2 = self.rp_reify_go(env, offset + 1, body);
                self.ctx.mk_lambda(binder_name, binder_style, d2, b2)
            }
            Pi { binder_name, binder_style, binder_type, body, .. } => {
                let d2 = self.rp_reify_go(env, offset, binder_type);
                let b2 = self.rp_reify_go(env, offset + 1, body);
                self.ctx.mk_pi(binder_name, binder_style, d2, b2)
            }
            Let { binder_name, binder_type, val, body, nondep, .. } => {
                let t2 = self.rp_reify_go(env, offset, binder_type);
                let v2 = self.rp_reify_go(env, offset, val);
                let b2 = self.rp_reify_go(env, offset + 1, body);
                self.ctx.mk_let(binder_name, t2, v2, b2, nondep)
            }
            Proj { ty_name, idx, structure, .. } => {
                let x2 = self.rp_reify_go(env, offset, structure);
                self.ctx.mk_proj(ty_name, idx, x2)
            }
            _ => e,
        };
        self.rp.reify_go_cache.insert(memo_key, r);
        r
    }

    // ---- eqMod: structural equality modulo substitution ----

    fn rp_clo_eq(&mut self, t: Clo<'t>, s: Clo<'t>) -> bool {
        self.rp_eq_mod(t.e, t.env, 0, s.e, s.env, 0)
    }

    fn rp_eq_mod(
        &mut self,
        ae: ExprPtr<'t>,
        aenv: EnvId,
        aoff: u16,
        be: ExprPtr<'t>,
        benv: EnvId,
        boff: u16,
    ) -> bool {
        // resolve a-side head indirections
        if let Var { dbj_idx: i, .. } = self.ctx.read_expr(ae) {
            if i >= aoff {
                return match self.rp_lookup(aenv, i - aoff) {
                    Entry::Val(e2, env2) => self.rp_eq_mod(e2, env2, 0, be, benv, boff),
                    Entry::Neu(fv) => self.rp_eq_mod_neu(fv, be, benv, boff),
                };
            }
        }
        // resolve b-side
        if let Var { dbj_idx: j, .. } = self.ctx.read_expr(be) {
            if j >= boff {
                return match self.rp_lookup(benv, j - boff) {
                    Entry::Val(e2, env2) => self.rp_eq_mod(ae, aenv, aoff, e2, env2, 0),
                    Entry::Neu(fv) => self.rp_eq_mod_neu_rev(ae, aenv, aoff, fv),
                };
            }
        }
        // fast paths
        if self.lbr(ae) <= aoff && self.lbr(be) <= boff && aoff == 0 && boff == 0 {
            return ae == be;
        }
        if ae == be && aenv == benv && aoff == boff {
            return true;
        }
        if ae == be && self.lbr(ae) <= aoff.min(boff) {
            return true;
        }
        let composite = matches!(
            self.ctx.read_expr(ae),
            App { .. } | Lambda { .. } | Pi { .. } | Let { .. } | Proj { .. }
        );
        let memo_key = (ae, aenv, aoff, be, benv, boff);
        if composite {
            if let Some(&r) = self.rp.eq_mod_cache.get(&memo_key) {
                return r;
            }
        }
        let (an, bn) = self.ctx.read_expr_pair(ae, be);
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
                self.rp_eq_mod(f1, aenv, aoff, f2, benv, boff)
                    && self.rp_eq_mod(a1, aenv, aoff, a2, benv, boff)
            }
            (
                Lambda { binder_type: d1, body: b1, .. },
                Lambda { binder_type: d2, body: b2, .. },
            )
            | (Pi { binder_type: d1, body: b1, .. }, Pi { binder_type: d2, body: b2, .. }) => {
                self.rp_eq_mod(d1, aenv, aoff, d2, benv, boff)
                    && self.rp_eq_mod(b1, aenv, aoff + 1, b2, benv, boff + 1)
            }
            (
                Let { binder_type: t1, val: v1, body: b1, .. },
                Let { binder_type: t2, val: v2, body: b2, .. },
            ) => {
                self.rp_eq_mod(t1, aenv, aoff, t2, benv, boff)
                    && self.rp_eq_mod(v1, aenv, aoff, v2, benv, boff)
                    && self.rp_eq_mod(b1, aenv, aoff + 1, b2, benv, boff + 1)
            }
            (
                Proj { ty_name: n1, idx: i1, structure: e1, .. },
                Proj { ty_name: n2, idx: i2, structure: e2, .. },
            ) => n1 == n2 && i1 == i2 && self.rp_eq_mod(e1, aenv, aoff, e2, benv, boff),
            _ => false,
        };
        if composite {
            self.rp.eq_mod_cache.insert(memo_key, r);
        }
        r
    }

    fn rp_eq_mod_neu(&mut self, fv: ExprPtr<'t>, be: ExprPtr<'t>, benv: EnvId, boff: u16) -> bool {
        if let Var { dbj_idx: j, .. } = self.ctx.read_expr(be) {
            if j >= boff {
                return match self.rp_lookup(benv, j - boff) {
                    Entry::Val(e2, env2) => self.rp_eq_mod_neu(fv, e2, env2, 0),
                    Entry::Neu(g) => fv == g,
                };
            }
            return false;
        }
        be == fv
    }

    fn rp_eq_mod_neu_rev(
        &mut self,
        ae: ExprPtr<'t>,
        aenv: EnvId,
        aoff: u16,
        fv: ExprPtr<'t>,
    ) -> bool {
        if let Var { dbj_idx: i, .. } = self.ctx.read_expr(ae) {
            if i >= aoff {
                return match self.rp_lookup(aenv, i - aoff) {
                    Entry::Val(e2, env2) => self.rp_eq_mod_neu_rev(e2, env2, 0, fv),
                    Entry::Neu(g) => g == fv,
                };
            }
            return false;
        }
        ae == fv
    }

    fn rp_s_quick_eq(&mut self, t: &SClo<'t>, s: &SClo<'t>) -> bool {
        if !self.rp_clo_eq(t.head, s.head) || t.spine.len() != s.spine.len() {
            return false;
        }
        for (&x, &y) in t.spine.iter().zip(s.spine.iter()) {
            if !self.rp_clo_eq(x, y) {
                return false;
            }
        }
        true
    }

    // ---- whnf ----

    pub(crate) fn rp_mk_sclo(&self, c: Clo<'t>) -> SClo<'t> {
        let mut spine = Vec::new();
        let mut e = c.e;
        while let App { fun, arg, .. } = self.ctx.read_expr(e) {
            spine.push(Clo { e: arg, env: c.env });
            e = fun;
        }
        spine.reverse();
        SClo { head: Clo { e, env: c.env }, spine }
    }

    pub(crate) fn rp_whnf_clo(&mut self, c: Clo<'t>) -> SClo<'t> {
        let c = self.rp_norm_clo(c);
        match self.ctx.read_expr(c.e) {
            NatLit { .. } | StringLit { .. } | Sort { .. } | Pi { .. } | Lambda { .. }
            | Local { .. } => return SClo { head: c, spine: Vec::new() },
            _ => {}
        }
        if let Some(r) = self.rp.whnf_cache.get(&c) {
            return r.clone();
        }
        let s = self.rp_mk_sclo(c);
        let r = self.rp_whnf(s);
        self.rp.whnf_cache.insert(c, r.clone());
        r
    }

    pub(crate) fn rp_whnf(&mut self, s: SClo<'t>) -> SClo<'t> {
        let mut t = self.rp_whnf_core(s);
        for _ in 0..100_000u32 {
            if let Some(t2) = self.rp_reduce_nat(&t) {
                t = self.rp_whnf_core(t2);
                continue;
            }
            match self.rp_unfold_definition(&t) {
                Some(t2) => t = self.rp_whnf_core(t2),
                None => return t,
            }
        }
        panic!("whnf: reduction fuel exhausted")
    }

    pub(crate) fn rp_whnf_core(&mut self, s: SClo<'t>) -> SClo<'t> {
        self.rp_whnf_core_ext(s, false, false)
    }

    /// `cheap`: skip recursor and projection reduction (the C++ kernel's
    /// cheap_rec/cheap_proj), so def-eq can compare stuck same-head
    /// applications structurally before forcing e.g. Nat.below towers.
    fn rp_whnf_core_ext(&mut self, s: SClo<'t>, cheap_rec: bool, cheap_proj: bool) -> SClo<'t> {
        let SClo { head, spine } = s;
        let mut e = head.e;
        let mut env = head.env;
        // reversed spine: last element is the innermost (next) argument
        let mut rsp: Vec<Clo<'t>> = spine.into_iter().rev().collect();
        loop {
            match self.ctx.read_expr(e) {
                App { fun, arg, .. } => {
                    rsp.push(Clo { e: arg, env });
                    e = fun;
                }
                Var { .. } => {
                    let (e2, env2) = self.rp_chase(e, env);
                    if e2 == e && env2 == env {
                        break;
                    }
                    e = e2;
                    env = env2;
                }
                Lambda { body, .. } => {
                    if let Some(arg) = rsp.pop() {
                        env = self.rp_push_entry(env, Entry::Val(arg.e, arg.env));
                        e = body;
                    } else {
                        break;
                    }
                }
                Let { val, body, .. } => {
                    env = self.rp_push_entry(env, Entry::Val(val, env));
                    e = body;
                }
                Proj { idx, structure, .. } => {
                    let sb = if cheap_proj {
                        let s0 = self.rp_mk_sclo(Clo { e: structure, env });
                        self.rp_whnf_core_ext(s0, cheap_rec, cheap_proj)
                    } else {
                        self.rp_whnf_clo(Clo { e: structure, env })
                    };
                    let sb = self.rp_lit_to_ctor(sb);
                    let mut stepped = false;
                    if let Const { name, .. } = self.ctx.read_expr(sb.head.e) {
                        if let Some(ctor) = self.env.get_constructor(&name) {
                            let num_params = ctor.num_params;
                            if let Some(&fld) = sb.spine.get(num_params as usize + idx) {
                                e = fld.e;
                                env = fld.env;
                                stepped = true;
                            }
                        }
                    }
                    if !stepped {
                        break;
                    }
                }
                Const { name, .. } => {
                    let nc = &self.ctx.export_file.name_cache;
                    let is_quot_op = Some(name) == nc.quot_lift || Some(name) == nc.quot_ind;
                    let reducible = match self.env.get_declar(&name) {
                        Some(Declar::Quot { .. }) => is_quot_op,
                        Some(Declar::Recursor(..)) => true,
                        _ => false,
                    };
                    if !reducible {
                        break;
                    }
                    let head = Clo { e, env };
                    if let Some(r) = self.rp_reduce_recursor(head, &rsp, cheap_rec, cheap_proj) {
                        e = r.head.e;
                        env = r.head.env;
                        rsp = r.spine;
                        rsp.reverse();
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        rsp.reverse();
        SClo { head: Clo { e, env }, spine: rsp }
    }

    /// Follow bvar -> Val chains to their base, with path compression: long
    /// chains (Nat.below towers) are resolved once and memoized per link.
    /// Neu entries resolve to the fvar expr. Returns the input unchanged only
    /// for non-bvar heads.
    fn rp_chase(&mut self, e0: ExprPtr<'t>, env0: EnvId) -> (ExprPtr<'t>, EnvId) {
        let mut e = e0;
        let mut env = env0;
        let mut trail: Vec<(ExprPtr<'t>, EnvId)> = Vec::new();
        loop {
            let i = match self.ctx.read_expr(e) {
                Var { dbj_idx, .. } => dbj_idx,
                _ => break,
            };
            if let Some(&r) = self.rp.chase_cache.get(&(e, env)) {
                e = r.0;
                env = r.1;
                break;
            }
            trail.push((e, env));
            match self.rp_lookup(env, i) {
                Entry::Neu(fv) => {
                    e = fv;
                    env = ENV_NIL;
                    break;
                }
                Entry::Val(e2, env2) => {
                    e = e2;
                    env = env2;
                }
            }
        }
        for key in trail {
            self.rp.chase_cache.insert(key, (e, env));
        }
        (e, env)
    }

    fn rp_lit_to_ctor(&mut self, s: SClo<'t>) -> SClo<'t> {
        if !s.spine.is_empty() {
            return s;
        }
        match self.ctx.read_expr(s.head.e) {
            NatLit { ptr, .. } => match self.ctx.nat_lit_to_constructor(ptr) {
                Some(e) => self.rp_mk_sclo(Clo::of(e)),
                None => s,
            },
            StringLit { ptr, .. } => match self.ctx.str_lit_to_constructor(ptr) {
                Some(e) => {
                    let m = self.rp_mk_sclo(Clo::of(e));
                    self.rp_whnf(m)
                }
                None => s,
            },
            _ => s,
        }
    }

    fn rp_delta_hint(&self, e: ExprPtr<'t>) -> Option<ReducibilityHint> {
        if let Const { name, .. } = self.ctx.read_expr(e) {
            match self.env.get_declar(&name) {
                Some(Declar::Definition { hint, .. }) => Some(*hint),
                Some(Declar::Theorem { .. }) => Some(ReducibilityHint::Opaque),
                _ => None,
            }
        } else {
            None
        }
    }

    fn rp_unfold_definition(&mut self, s: &SClo<'t>) -> Option<SClo<'t>> {
        let (name, levels) = match self.ctx.read_expr(s.head.e) {
            Const { name, levels, .. } => (name, levels),
            _ => return None,
        };
        let (def_uparams, def_value) = self.env.get_declar_val(&name)?;
        if self.ctx.read_levels(levels).len() != self.ctx.read_levels(def_uparams).len() {
            return None;
        }
        let r = self.ctx.subst_expr_levels(def_value, def_uparams, levels);
        let m = self.rp_mk_sclo(Clo::of(r));
        let mut spine = m.spine;
        spine.extend_from_slice(&s.spine);
        Some(SClo { head: m.head, spine })
    }

    fn rp_nat_lit(&mut self, c: Clo<'t>) -> Option<BigUint> {
        let nc = self.ctx.export_file.name_cache;
        match self.ctx.read_expr(c.e) {
            NatLit { ptr, .. } => return self.ctx.read_bignum(ptr).cloned(),
            Const { name, .. } if Some(name) == nc.nat_zero => return Some(BigUint::zero()),
            _ => {}
        }
        let w = self.rp_whnf_clo(c);
        if !w.spine.is_empty() {
            return None;
        }
        match self.ctx.read_expr(w.head.e) {
            NatLit { ptr, .. } => self.ctx.read_bignum(ptr).cloned(),
            Const { name, .. } if Some(name) == nc.nat_zero => Some(BigUint::zero()),
            _ => None,
        }
    }

    fn rp_mk_nat_sclo(&mut self, n: BigUint) -> SClo<'t> {
        let e = self.ctx.mk_nat_lit_quick(n).unwrap();
        SClo { head: Clo::of(e), spine: Vec::new() }
    }

    fn rp_mk_bool_sclo(&mut self, b: bool) -> Option<SClo<'t>> {
        let e = self.ctx.bool_to_expr(b)?;
        Some(SClo { head: Clo::of(e), spine: Vec::new() })
    }

    fn rp_reduce_nat(&mut self, s: &SClo<'t>) -> Option<SClo<'t>> {
        if !self.ctx.export_file.config.nat_extension {
            return None;
        }
        let f = match self.ctx.read_expr(s.head.e) {
            Const { name, levels, .. } if self.ctx.read_levels(levels).is_empty() => name,
            _ => return None,
        };
        let nc = self.ctx.export_file.name_cache;
        if s.spine.len() == 1 {
            if Some(f) == nc.nat_succ {
                if let Some(v) = self.rp_nat_lit(s.spine[0]) {
                    return Some(self.rp_mk_nat_sclo(v + 1u32));
                }
            }
            return None;
        }
        if s.spine.len() != 2 {
            return None;
        }
        macro_rules! bin {
            ($op:expr) => {{
                let v1 = self.rp_nat_lit(s.spine[0])?;
                let v2 = self.rp_nat_lit(s.spine[1])?;
                #[allow(clippy::redundant_closure_call)]
                let r: BigUint = ($op)(v1, v2);
                return Some(self.rp_mk_nat_sclo(r));
            }};
        }
        macro_rules! pred {
            ($op:expr) => {{
                let v1 = self.rp_nat_lit(s.spine[0])?;
                let v2 = self.rp_nat_lit(s.spine[1])?;
                #[allow(clippy::redundant_closure_call)]
                let r: bool = ($op)(&v1, &v2);
                return self.rp_mk_bool_sclo(r);
            }};
        }
        if Some(f) == nc.nat_add {
            bin!(|a: BigUint, b: BigUint| a + b)
        } else if Some(f) == nc.nat_sub {
            bin!(nat_sub)
        } else if Some(f) == nc.nat_mul {
            bin!(|a: BigUint, b: BigUint| a * b)
        } else if Some(f) == nc.nat_pow {
            let v2 = self.rp_nat_lit(s.spine[1])?;
            if v2 > BigUint::from(1u32 << 24) {
                return None;
            }
            let v1 = self.rp_nat_lit(s.spine[0])?;
            let r = num_traits::pow::Pow::pow(v1, v2.to_u32().unwrap());
            Some(self.rp_mk_nat_sclo(r))
        } else if Some(f) == nc.nat_gcd {
            bin!(|a: BigUint, b: BigUint| nat_gcd(&a, &b))
        } else if Some(f) == nc.nat_mod {
            bin!(nat_mod)
        } else if Some(f) == nc.nat_div {
            bin!(nat_div)
        } else if Some(f) == nc.nat_beq {
            pred!(|a: &BigUint, b: &BigUint| a == b)
        } else if Some(f) == nc.nat_ble {
            pred!(|a: &BigUint, b: &BigUint| a <= b)
        } else if Some(f) == nc.nat_land {
            bin!(nat_land)
        } else if Some(f) == nc.nat_lor {
            bin!(nat_lor)
        } else if Some(f) == nc.nat_xor {
            bin!(|a: BigUint, b: BigUint| nat_xor(&a, &b))
        } else if Some(f) == nc.nat_shl {
            let v2 = self.rp_nat_lit(s.spine[1])?;
            if v2 > BigUint::from(1u32 << 24) {
                return None;
            }
            let v1 = self.rp_nat_lit(s.spine[0])?;
            let r = v1 << v2.to_u64().unwrap();
            Some(self.rp_mk_nat_sclo(r))
        } else if Some(f) == nc.nat_shr {
            let v2 = self.rp_nat_lit(s.spine[1])?;
            let v1 = self.rp_nat_lit(s.spine[0])?;
            let r = match v2.to_u64() {
                Some(sh) => v1 >> sh,
                None => BigUint::zero(),
            };
            Some(self.rp_mk_nat_sclo(r))
        } else {
            None
        }
    }

    /// `rsp` is the reversed spine (last = innermost arg). Returns the reduct
    /// with a normal-order spine, or None; allocates only on success.
    /// `cheap`: the major premise is reduced with cheap whnf_core only, and
    /// the K/struct constructor conversions (which infer types) are skipped;
    /// the full pass at def-eq's last resort covers them.
    fn rp_reduce_recursor(
        &mut self,
        head: Clo<'t>,
        rsp: &[Clo<'t>],
        cheap_rec: bool,
        cheap_proj: bool,
    ) -> Option<SClo<'t>> {
        let (fname, ls) = match self.ctx.read_expr(head.e) {
            Const { name, levels, .. } => (name, levels),
            _ => return None,
        };
        let len = rsp.len();
        let idx = |i: usize| rsp[len - 1 - i];
        // quot
        if matches!(self.env.get_declar(&fname), Some(Declar::Quot { .. })) {
            let nc = self.ctx.export_file.name_cache;
            let pos = if Some(fname) == nc.quot_lift {
                Some((5usize, 3usize))
            } else if Some(fname) == nc.quot_ind {
                Some((4, 3))
            } else {
                None
            };
            if let Some((mk_pos, arg_pos)) = pos {
                if mk_pos < len {
                    let mk = if cheap_rec {
                        let s0 = self.rp_mk_sclo(idx(mk_pos));
                        self.rp_whnf_core_ext(s0, cheap_rec, cheap_proj)
                    } else {
                        self.rp_whnf_clo(idx(mk_pos))
                    };
                    let is_mk = matches!(self.ctx.read_expr(mk.head.e),
                        Const { name, .. } if Some(name) == nc.quot_mk)
                        && mk.spine.len() == 3;
                    if is_mk {
                        let m = self.rp_mk_sclo(idx(arg_pos));
                        let mut spine = m.spine;
                        spine.push(mk.spine[2]);
                        spine.extend((mk_pos + 1..len).map(idx));
                        return Some(SClo { head: m.head, spine });
                    }
                }
                return None;
            }
            return None;
        }
        // iota
        let rec = self.env.get_recursor(&fname)?;
        if self.ctx.read_levels(rec.info.uparams).len() != self.ctx.read_levels(ls).len() {
            return None;
        }
        let (num_params, num_motives, num_minors) = (rec.num_params, rec.num_motives, rec.num_minors);
        let is_k = rec.is_k;
        let major_idx = rec.major_idx();
        let rec_uparams = rec.info.uparams;
        let rec_rules = rec.rec_rules.clone();
        let major_induct = self.ctx.get_major_induct(rec);
        if major_idx >= len {
            return None;
        }
        let mut mj = if cheap_rec {
            let s0 = self.rp_mk_sclo(idx(major_idx));
            self.rp_whnf_core_ext(s0, cheap_rec, cheap_proj)
        } else {
            self.rp_whnf_clo(idx(major_idx))
        };
        if is_k && !cheap_rec {
            if let Some(mi) = major_induct {
                mj = self.rp_to_ctor_when_k(num_params, mi, mj);
            }
        }
        mj = self.rp_lit_to_ctor(mj);
        if !cheap_rec {
            if let Some(mi) = major_induct {
                mj = self.rp_to_ctor_when_struct(mi, mj);
            }
        }
        let ctor = match self.ctx.read_expr(mj.head.e) {
            Const { name, .. } => name,
            _ => return None,
        };
        let rule = rec_rules.iter().find(|r| r.ctor_name == ctor)?;
        let rule_nfields = rule.ctor_telescope_size_wo_params;
        if (rule_nfields as usize) > mj.spine.len() {
            return None;
        }
        let rhs = self.ctx.subst_expr_levels(rule.val, rec_uparams, ls);
        let first_index_idx = (num_params + num_motives + num_minors) as usize;
        let m = self.rp_mk_sclo(Clo::of(rhs));
        let mut spine = m.spine;
        spine.extend((0..first_index_idx).map(idx));
        spine.extend_from_slice(&mj.spine[mj.spine.len() - rule_nfields as usize..]);
        spine.extend((major_idx + 1..len).map(idx));
        Some(SClo { head: m.head, spine })
    }

    fn rp_to_ctor_when_k(
        &mut self,
        num_params: u16,
        major_induct: NamePtr<'t>,
        mj: SClo<'t>,
    ) -> SClo<'t> {
        let ty = self.rp_infer_s(&mj, InferOnly);
        let app_type = self.rp_whnf_clo(Clo::of(ty));
        let (i, ils) = match self.ctx.read_expr(app_type.head.e) {
            Const { name, levels, .. } => (name, levels),
            _ => return mj,
        };
        if i != major_induct {
            return mj;
        }
        let ctor = match self.env.get_inductive(&i) {
            Some(ind) => match ind.all_ctor_names.first() {
                Some(&c) => c,
                None => return mj,
            },
            None => return mj,
        };
        let ctor_e = self.ctx.mk_const(ctor, ils);
        let new_ctor = SClo {
            head: Clo::of(ctor_e),
            spine: app_type.spine[..(num_params as usize).min(app_type.spine.len())].to_vec(),
        };
        let new_ty = self.rp_infer_s(&new_ctor, InferOnly);
        let app_clo = self.rp_sclo_as_clo(&app_type);
        if self.rp_is_def_eq(app_clo, Clo::of(new_ty)) {
            return new_ctor;
        }
        mj
    }

    fn rp_to_ctor_when_struct(&mut self, induct: NamePtr<'t>, mj: SClo<'t>) -> SClo<'t> {
        if !self.env.can_be_struct(&induct) {
            return mj;
        }
        if let Const { name, .. } = self.ctx.read_expr(mj.head.e) {
            if self.env.get_constructor(&name).is_some() {
                return mj;
            }
        }
        let ty = self.rp_infer_s(&mj, InferOnly);
        let e_type = self.rp_whnf_clo(Clo::of(ty));
        let (i, ils) = match self.ctx.read_expr(e_type.head.e) {
            Const { name, levels, .. } => (name, levels),
            _ => return mj,
        };
        if i != induct {
            return mj;
        }
        // not for propositions
        let tyty = self.rp_infer_s(&e_type, InferOnly);
        let tyty_w = self.rp_whnf_clo(Clo::of(tyty));
        if let Sort { level, .. } = self.ctx.read_expr(tyty_w.head.e) {
            if tyty_w.spine.is_empty() && self.ctx.is_zero(level) {
                return mj;
            }
        }
        let ctor = match self.env.get_inductive(&i) {
            Some(ind) => match ind.all_ctor_names.first() {
                Some(&c) => c,
                None => return mj,
            },
            None => return mj,
        };
        let (num_params, num_fields) = match self.env.get_constructor(&ctor) {
            Some(cd) => (cd.num_params, cd.num_fields),
            None => return mj,
        };
        let mj_clo = self.rp_sclo_as_clo(&mj);
        let mut spine: Vec<Clo<'t>> =
            e_type.spine[..(num_params as usize).min(e_type.spine.len())].to_vec();
        for fi in 0..num_fields {
            let bv = self.ctx.mk_var(0);
            let proj = self.ctx.mk_proj(i, fi as usize, bv);
            let env = self.rp_push_entry(ENV_NIL, Entry::Val(mj_clo.e, mj_clo.env));
            spine.push(Clo { e: proj, env });
        }
        let head = self.ctx.mk_const(ctor, ils);
        SClo { head: Clo::of(head), spine }
    }

    /// Closure denoting the same term as `s` (no lifting; head and spine go
    /// through a fresh environment of bvars). Unstable env identity.
    fn rp_sclo_as_clo(&mut self, s: &SClo<'t>) -> Clo<'t> {
        if s.spine.is_empty() {
            return s.head;
        }
        let mut env = self.rp_push_entry(ENV_NIL, Entry::Val(s.head.e, s.head.env));
        let mut e = self.ctx.mk_var(u16::try_from(s.spine.len()).unwrap());
        for (i, arg) in s.spine.iter().enumerate() {
            env = self.rp_push_entry(env, Entry::Val(arg.e, arg.env));
            let bv = self.ctx.mk_var(u16::try_from(s.spine.len() - 1 - i).unwrap());
            e = self.ctx.mk_app(e, bv);
        }
        Clo { e, env }
    }

    // ---- def-eq ----

    pub(crate) fn rp_is_def_eq(&mut self, t: Clo<'t>, s: Clo<'t>) -> bool {
        if self.rp_clo_eq(t, s) {
            return true;
        }
        let (tk, sk) = (self.rp_norm_clo(t), self.rp_norm_clo(s));
        let pk = if clo_le(&tk, &sk) { (tk, sk) } else { (sk, tk) };
        if self.rp.eq_pos.contains(&pk) {
            return true;
        }
        if self.rp.eq_neg.contains(&pk) {
            return false;
        }
        let ts = self.rp_mk_sclo(t);
        let ss = self.rp_mk_sclo(s);
        let r = self.rp_is_def_eq_s(ts, ss);
        if r {
            self.rp.eq_pos.insert(pk);
        } else {
            self.rp.eq_neg.insert(pk);
        }
        r
    }

    fn rp_sclo_key(&self, s: &SClo<'t>) -> SKey<'t> {
        (self.rp_key(s.head), s.spine.iter().map(|&c| self.rp_key(c)).collect())
    }

    fn rp_is_def_eq_s(&mut self, t: SClo<'t>, s: SClo<'t>) -> bool {
        let tn = self.rp_whnf_core_ext(t, false, false);
        let sn = self.rp_whnf_core_ext(s, false, false);
        if self.rp_s_quick_eq(&tn, &sn) {
            return true;
        }
        let (k1, k2) = (self.rp_sclo_key(&tn), self.rp_sclo_key(&sn));
        let pk = if skey_le(&k1, &k2) { (k1, k2) } else { (k2, k1) };
        if self.rp.eq_s_pos.contains(&pk) {
            return true;
        }
        if self.rp.eq_s_neg.contains(&pk) {
            return false;
        }
        let r = self.rp_is_def_eq_s_core(tn, sn);
        if r {
            self.rp.eq_s_pos.insert(pk);
        } else {
            self.rp.eq_s_neg.insert(pk);
        }
        r
    }

    fn rp_is_def_eq_s_core(&mut self, tn: SClo<'t>, sn: SClo<'t>) -> bool {
        if let Some(b) = self.rp_quick_heads(&tn, &sn) {
            return b;
        }
        if let Some(b) = self.rp_def_eq_offset(&tn, &sn) {
            return b;
        }
        // proof irrelevance
        let t_ty = self.rp_infer_s(&tn, InferOnly);
        if self.rp_is_prop(t_ty) {
            let s_ty = self.rp_infer_s(&sn, InferOnly);
            if self.rp_is_def_eq(Clo::of(t_ty), Clo::of(s_ty)) {
                return true;
            }
        }
        match self.rp_lazy_delta(tn, sn, 10_000) {
            Ok(b) => b,
            Err((tn, sn)) => {
                if let Some(b) = self.rp_quick_heads(&tn, &sn) {
                    return b;
                }
                match self.ctx.read_expr_pair(tn.head.e, sn.head.e) {
                    (Const { name: c1, levels: l1, .. }, Const { name: c2, levels: l2, .. }) => {
                        if c1 == c2 && self.rp_lvl_eq_list(l1, l2) && self.rp_def_eq_spines(&tn, &sn)
                        {
                            return true;
                        }
                    }
                    (
                        Local { id: i1, binder_type: t1, .. },
                        Local { id: i2, binder_type: t2, .. },
                    ) => {
                        let heads_eq = tn.head.e == sn.head.e
                            || (i1 == i2 && self.rp_is_def_eq(Clo::of(t1), Clo::of(t2)));
                        if heads_eq && self.rp_def_eq_spines(&tn, &sn) {
                            return true;
                        }
                    }
                    (
                        Proj { idx: i1, structure: e1, .. },
                        Proj { idx: i2, structure: e2, .. },
                    ) => {
                        if i1 == i2
                            && self.rp_is_def_eq(
                                Clo { e: e1, env: tn.head.env },
                                Clo { e: e2, env: sn.head.env },
                            )
                            && self.rp_def_eq_spines(&tn, &sn)
                        {
                            return true;
                        }
                    }
                    _ => {}
                }
                if self.rp_try_eta_expansion(&tn, &sn) {
                    return true;
                }
                if self.rp_try_eta_struct(&tn, &sn) {
                    return true;
                }
                if self.rp_def_eq_string_lit(&tn, &sn) {
                    return true;
                }
                if self.rp_def_eq_unit_like(&tn, &sn) {
                    return true;
                }
                // last resort: reduce fully (iota and proj) and retry if
                // either side changed, as in the C++ kernel's final
                // whnf_core-and-recurse step
                let tf = self.rp_whnf_core(tn.clone());
                let sf = self.rp_whnf_core(sn.clone());
                let t_changed = !(tf.head == tn.head && tf.spine == tn.spine);
                let s_changed = !(sf.head == sn.head && sf.spine == sn.spine);
                if t_changed || s_changed {
                    return self.rp_is_def_eq_s(tf, sf);
                }
                false
            }
        }
    }

    fn rp_lvl_eq1(&mut self, l1: LevelPtr<'t>, l2: LevelPtr<'t>) -> bool {
        if l1 == l2 {
            return true;
        }
        let k = if l1.get_hash() <= l2.get_hash() { (l1, l2) } else { (l2, l1) };
        if let Some(&r) = self.rp.lvl_eq_cache.get(&k) {
            return r;
        }
        let r = self.ctx.eq_antisymm(l1, l2);
        self.rp.lvl_eq_cache.insert(k, r);
        r
    }

    fn rp_lvl_eq_list(&mut self, ls: LevelsPtr<'t>, rs: LevelsPtr<'t>) -> bool {
        if ls == rs {
            return true;
        }
        let (xs, ys) = (self.ctx.read_levels(ls), self.ctx.read_levels(rs));
        if xs.len() != ys.len() {
            return false;
        }
        for (&a, &b) in xs.iter().zip(ys.iter()) {
            if !self.rp_lvl_eq1(a, b) {
                return false;
            }
        }
        true
    }

    fn rp_def_eq_spines(&mut self, t: &SClo<'t>, s: &SClo<'t>) -> bool {
        if t.spine.len() != s.spine.len() {
            return false;
        }
        for (&x, &y) in t.spine.iter().zip(s.spine.iter()) {
            if !self.rp_is_def_eq(x, y) {
                return false;
            }
        }
        true
    }

    fn rp_quick_heads(&mut self, tn: &SClo<'t>, sn: &SClo<'t>) -> Option<bool> {
        match self.ctx.read_expr_pair(tn.head.e, sn.head.e) {
            (Lambda { .. }, Lambda { .. }) | (Pi { .. }, Pi { .. }) => {
                if tn.spine.is_empty() && sn.spine.is_empty() {
                    return Some(self.rp_def_eq_binding(tn.head, sn.head));
                }
                None
            }
            (Sort { level: l1, .. }, Sort { level: l2, .. }) => Some(self.rp_lvl_eq1(l1, l2)),
            (NatLit { ptr: a, .. }, NatLit { ptr: b, .. }) => {
                if tn.spine.is_empty() && sn.spine.is_empty() {
                    return Some(a == b);
                }
                None
            }
            (StringLit { ptr: a, .. }, StringLit { ptr: b, .. }) => {
                if tn.spine.is_empty() && sn.spine.is_empty() {
                    return Some(a == b);
                }
                None
            }
            _ => None,
        }
    }

    fn rp_def_eq_binding(&mut self, t: Clo<'t>, s: Clo<'t>) -> bool {
        let (td, tb, sd, sb) = match self.ctx.read_expr_pair(t.e, s.e) {
            (
                Lambda { binder_type: td, body: tb, .. },
                Lambda { binder_type: sd, body: sb, .. },
            )
            | (Pi { binder_type: td, body: tb, .. }, Pi { binder_type: sd, body: sb, .. }) => {
                (td, tb, sd, sb)
            }
            _ => return self.rp_is_def_eq(t, s),
        };
        let tdc = Clo { e: td, env: t.env };
        let sdc = Clo { e: sd, env: s.env };
        if !self.rp_is_def_eq(tdc, sdc) {
            return false;
        }
        let d = self.rp_reify(sdc);
        let fv = self.rp_fresh_fvar(d);
        let tenv = self.rp_push_entry(t.env, Entry::Neu(fv));
        let senv = self.rp_push_entry(s.env, Entry::Neu(fv));
        self.rp_def_eq_binding(Clo { e: tb, env: tenv }, Clo { e: sb, env: senv })
    }

    /// Whether the substitution instance denoted by `(e, env)` contains a
    /// free variable: an fvar node in `e` itself, or an occurring loose bvar
    /// of `e` resolving through `env` to a neutral entry or to a value
    /// closure that contains one. Follows the same traversal as `reify_go`,
    /// so the answer equals `has_fvars` of the reification.
    fn rp_clo_has_fvar(&mut self, e: ExprPtr<'t>, env: EnvId) -> bool {
        self.rp_clo_has_fvar_go(e, env, 0)
    }

    fn rp_clo_has_fvar_go(&mut self, e: ExprPtr<'t>, env: EnvId, offset: u16) -> bool {
        if self.ctx.has_fvars(e) {
            return true;
        }
        if self.lbr(e) <= offset || env == ENV_NIL {
            return false;
        }
        let key = (e, env, offset);
        if let Some(&r) = self.rp.clo_fvar_cache.get(&key) {
            return r;
        }
        let r = match self.ctx.read_expr(e) {
            Var { dbj_idx, .. } => match self.rp_lookup(env, dbj_idx - offset) {
                Entry::Neu(_) => true,
                Entry::Val(e2, env2) => self.rp_clo_has_fvar_go(e2, env2, 0),
            },
            App { fun, arg, .. } => {
                self.rp_clo_has_fvar_go(fun, env, offset)
                    || self.rp_clo_has_fvar_go(arg, env, offset)
            }
            Lambda { binder_type, body, .. } | Pi { binder_type, body, .. } => {
                self.rp_clo_has_fvar_go(binder_type, env, offset)
                    || self.rp_clo_has_fvar_go(body, env, offset + 1)
            }
            Let { binder_type, val, body, .. } => {
                self.rp_clo_has_fvar_go(binder_type, env, offset)
                    || self.rp_clo_has_fvar_go(val, env, offset)
                    || self.rp_clo_has_fvar_go(body, env, offset + 1)
            }
            Proj { structure, .. } => self.rp_clo_has_fvar_go(structure, env, offset),
            _ => false,
        };
        self.rp.clo_fvar_cache.insert(key, r);
        r
    }

    fn rp_sclo_has_fvar(&mut self, s: &SClo<'t>) -> bool {
        if self.rp_clo_has_fvar(s.head.e, s.head.env) {
            return true;
        }
        for &c in s.spine.iter() {
            if self.rp_clo_has_fvar(c.e, c.env) {
                return true;
            }
        }
        false
    }

    /// Would `reduce_nat` even dispatch on this head? Cheap pre-test so the
    /// fvar-occurrence probe only runs where a reduction could fire.
    fn rp_nat_op_head(&self, s: &SClo<'t>) -> bool {
        let f = match self.ctx.read_expr(s.head.e) {
            Const { name, levels, .. } if self.ctx.read_levels(levels).is_empty() => name,
            _ => return false,
        };
        let nc = &self.ctx.export_file.name_cache;
        let f = Some(f);
        match s.spine.len() {
            1 => f == nc.nat_succ,
            2 => {
                f == nc.nat_add
                    || f == nc.nat_sub
                    || f == nc.nat_mul
                    || f == nc.nat_pow
                    || f == nc.nat_gcd
                    || f == nc.nat_mod
                    || f == nc.nat_div
                    || f == nc.nat_beq
                    || f == nc.nat_ble
                    || f == nc.nat_land
                    || f == nc.nat_lor
                    || f == nc.nat_xor
                    || f == nc.nat_shl
                    || f == nc.nat_shr
            }
            _ => false,
        }
    }

    fn rp_nat_zero_s(&mut self, t: &SClo<'t>) -> bool {
        t.spine.is_empty() && self.ctx.is_nat_zero(t.head.e)
    }

    fn rp_nat_succ_of(&mut self, t: &SClo<'t>) -> Option<SClo<'t>> {
        if t.spine.is_empty() {
            if let NatLit { ptr, .. } = self.ctx.read_expr(t.head.e) {
                let n = self.ctx.read_bignum(ptr).cloned()?;
                if !n.is_zero() {
                    return Some(self.rp_mk_nat_sclo(n - 1u32));
                }
            }
            None
        } else if t.spine.len() == 1
            && matches!(self.ctx.read_expr(t.head.e),
                Const { name, .. } if Some(name) == self.ctx.export_file.name_cache.nat_succ)
        {
            Some(self.rp_mk_sclo(t.spine[0]))
        } else {
            None
        }
    }

    fn rp_def_eq_offset(&mut self, t: &SClo<'t>, s: &SClo<'t>) -> Option<bool> {
        if self.rp_nat_zero_s(t) && self.rp_nat_zero_s(s) {
            return Some(true);
        }
        if let (Some(t2), Some(s2)) = (self.rp_nat_succ_of(t), self.rp_nat_succ_of(s)) {
            return Some(self.rp_is_def_eq_s(t2, s2));
        }
        None
    }

    /// `Ok(b)` decided; `Err((tn, sn))` both irreducible.
    #[allow(clippy::type_complexity)]
    fn rp_lazy_delta(
        &mut self,
        tn: SClo<'t>,
        sn: SClo<'t>,
        fuel: u32,
    ) -> Result<bool, (SClo<'t>, SClo<'t>)> {
        if fuel == 0 {
            panic!("lazy_delta: fuel exhausted");
        }
        if self.rp_s_quick_eq(&tn, &sn) {
            return Ok(true);
        }
        if let Some(b) = self.rp_def_eq_offset(&tn, &sn) {
            return Ok(b);
        }
        if (self.rp_nat_op_head(&tn) || self.rp_nat_op_head(&sn))
            && !self.rp_sclo_has_fvar(&tn)
            && !self.rp_sclo_has_fvar(&sn)
        {
            if let Some(t2) = self.rp_reduce_nat(&tn) {
                return Ok(self.rp_is_def_eq_s(t2, sn));
            }
            if let Some(s2) = self.rp_reduce_nat(&sn) {
                return Ok(self.rp_is_def_eq_s(tn, s2));
            }
        }
        self.rp_check_native(&tn);
        self.rp_check_native(&sn);
        let dt = self.rp_delta_hint(tn.head.e);
        let ds = self.rp_delta_hint(sn.head.e);
        match (dt, ds) {
            (None, None) => Err((tn, sn)),
            (Some(_), None) => {
                if let Some(s2) = self.rp_try_unfold_proj_app(&sn) {
                    return self.rp_lazy_delta(tn, s2, fuel - 1);
                }
                let t2 = self.rp_delta1(&tn);
                self.rp_lazy_delta(t2, sn, fuel - 1)
            }
            (None, Some(_)) => {
                if let Some(t2) = self.rp_try_unfold_proj_app(&tn) {
                    return self.rp_lazy_delta(t2, sn, fuel - 1);
                }
                let s2 = self.rp_delta1(&sn);
                self.rp_lazy_delta(tn, s2, fuel - 1)
            }
            (Some(ht), Some(hs)) => {
                if ht.is_lt(&hs) {
                    let s2 = self.rp_delta1(&sn);
                    self.rp_lazy_delta(tn, s2, fuel - 1)
                } else if hs.is_lt(&ht) {
                    let t2 = self.rp_delta1(&tn);
                    self.rp_lazy_delta(t2, sn, fuel - 1)
                } else {
                    // same head, regular hints: try spine comparison
                    if !tn.spine.is_empty() && !sn.spine.is_empty() {
                        let same_const =
                            match self.ctx.read_expr_pair(tn.head.e, sn.head.e) {
                                (
                                    Const { name: c1, levels: l1, .. },
                                    Const { name: c2, levels: l2, .. },
                                ) => {
                                    if c1 == c2 && matches!(ht, ReducibilityHint::Regular(_)) {
                                        self.rp_lvl_eq_list(l1, l2)
                                    } else {
                                        false
                                    }
                                }
                                _ => false,
                            };
                        if same_const && self.rp_def_eq_spines(&tn, &sn) {
                            return Ok(true);
                        }
                    }
                    let t2 = self.rp_delta1(&tn);
                    let s2 = self.rp_delta1(&sn);
                    self.rp_lazy_delta(t2, s2, fuel - 1)
                }
            }
        }
    }

    /// If the head is a projection, resolve it by full whnf_core; the
    /// baseline's tryUnfoldProjApp in the one-sided lazy-delta cases.
    fn rp_try_unfold_proj_app(&mut self, s: &SClo<'t>) -> Option<SClo<'t>> {
        if !matches!(self.ctx.read_expr(s.head.e), Proj { .. }) {
            return None;
        }
        let r = self.rp_whnf_core(s.clone());
        if r.head == s.head && r.spine == s.spine {
            None
        } else {
            Some(r)
        }
    }

    fn rp_delta1(&mut self, x: &SClo<'t>) -> SClo<'t> {
        let u = self.rp_unfold_definition(x).expect("delta on non-unfoldable");
        self.rp_whnf_core_ext(u, false, true)
    }

    fn rp_check_native(&self, s: &SClo<'t>) {
        if let Const { name, .. } = self.ctx.read_expr(s.head.e) {
            let nc = &self.ctx.export_file.name_cache;
            if Some(name) == nc.reduce_bool || Some(name) == nc.reduce_nat {
                panic!("native reduction (Lean.reduceBool/Lean.reduceNat) not supported");
            }
        }
    }

    fn rp_try_eta_expansion(&mut self, t: &SClo<'t>, s: &SClo<'t>) -> bool {
        if self.rp_try_eta_core(t, s) {
            return true;
        }
        self.rp_try_eta_core(s, t)
    }

    fn rp_try_eta_core(&mut self, t: &SClo<'t>, s: &SClo<'t>) -> bool {
        let t_is_lam =
            matches!(self.ctx.read_expr(t.head.e), Lambda { .. }) && t.spine.is_empty();
        let s_is_lam =
            matches!(self.ctx.read_expr(s.head.e), Lambda { .. }) && s.spine.is_empty();
        if !t_is_lam || s_is_lam {
            return false;
        }
        let s_ty = self.rp_infer_s(s, InferOnly);
        let s_ty_w = self.rp_whnf_clo(Clo::of(s_ty));
        if !s_ty_w.spine.is_empty() {
            return false;
        }
        let Pi { binder_type: dom, .. } = self.ctx.read_expr(s_ty_w.head.e) else {
            return false;
        };
        let d = self.rp_reify(Clo { e: dom, env: s_ty_w.head.env });
        let fv = self.rp_fresh_fvar(d);
        let Lambda { body: t_body, .. } = self.ctx.read_expr(t.head.e) else {
            return false;
        };
        let tenv = self.rp_push_entry(t.head.env, Entry::Neu(fv));
        let mut s_applied = s.clone();
        s_applied.spine.push(Clo::of(fv));
        let lhs = self.rp_mk_sclo(Clo { e: t_body, env: tenv });
        self.rp_is_def_eq_s(lhs, s_applied)
    }

    fn rp_try_eta_struct(&mut self, t: &SClo<'t>, s: &SClo<'t>) -> bool {
        if self.rp_try_eta_struct_core(t, s) {
            return true;
        }
        self.rp_try_eta_struct_core(s, t)
    }

    fn rp_try_eta_struct_core(&mut self, t: &SClo<'t>, s: &SClo<'t>) -> bool {
        let ctor = match self.ctx.read_expr(s.head.e) {
            Const { name, .. } => name,
            _ => return false,
        };
        let (induct, num_params, num_fields) = match self.env.get_constructor(&ctor) {
            Some(cd) => (cd.inductive_name, cd.num_params, cd.num_fields),
            None => return false,
        };
        if s.spine.len() != (num_params + num_fields) as usize {
            return false;
        }
        if !self.env.can_be_struct(&induct) {
            return false;
        }
        let t_ty = self.rp_infer_s(t, InferOnly);
        let s_ty = self.rp_infer_s(s, InferOnly);
        if !self.rp_is_def_eq(Clo::of(t_ty), Clo::of(s_ty)) {
            return false;
        }
        let t_clo = self.rp_sclo_as_clo(t);
        for i in num_params..(num_params + num_fields) {
            let bv = self.ctx.mk_var(0);
            let proj = self.ctx.mk_proj(induct, (i - num_params) as usize, bv);
            let env = self.rp_push_entry(ENV_NIL, Entry::Val(t_clo.e, t_clo.env));
            if !self.rp_is_def_eq(Clo { e: proj, env }, s.spine[i as usize]) {
                return false;
            }
        }
        true
    }

    fn rp_def_eq_string_lit(&mut self, t: &SClo<'t>, s: &SClo<'t>) -> bool {
        fn core<'x, 't: 'x, 'p: 't>(
            ck: &mut TypeChecker<'x, 't, 'p>,
            t: &SClo<'t>,
            s: &SClo<'t>,
        ) -> bool {
            let sp = match ck.ctx.read_expr(t.head.e) {
                StringLit { ptr, .. } => ptr,
                _ => return false,
            };
            let nc = &ck.ctx.export_file.name_cache;
            let s_head_ok = matches!(ck.ctx.read_expr(s.head.e),
                Const { name, .. } if Some(name) == nc.string_of_list || Some(name) == nc.string_mk);
            if !s_head_ok {
                return false;
            }
            let Some(e) = ck.ctx.str_lit_to_constructor(sp) else {
                return false;
            };
            let lhs = ck.rp_mk_sclo(Clo::of(e));
            ck.rp_is_def_eq_s(lhs, s.clone())
        }
        if core(self, t, s) {
            return true;
        }
        core(self, s, t)
    }

    fn rp_def_eq_unit_like(&mut self, t: &SClo<'t>, s: &SClo<'t>) -> bool {
        let t_ty = self.rp_infer_s(t, InferOnly);
        let ty_w = self.rp_whnf_clo(Clo::of(t_ty));
        let i = match self.ctx.read_expr(ty_w.head.e) {
            Const { name, .. } => name,
            _ => return false,
        };
        if !self.env.can_be_struct(&i) {
            return false;
        }
        let ctor0 = match self.env.get_inductive(&i) {
            Some(ind) => match ind.all_ctor_names.first() {
                Some(&c) => c,
                None => return false,
            },
            None => return false,
        };
        match self.env.get_constructor(&ctor0) {
            Some(cd) if cd.num_fields == 0 => {}
            _ => return false,
        }
        let ty_clo = self.rp_sclo_as_clo(&ty_w);
        let s_ty = self.rp_infer_s(s, InferOnly);
        self.rp_is_def_eq(ty_clo, Clo::of(s_ty))
    }

    /// Is `ty` a proposition, i.e. `ty : Prop`?
    fn rp_is_prop(&mut self, ty: ExprPtr<'t>) -> bool {
        let sort = self.rp_infer(Clo::of(ty), InferOnly);
        let w = self.rp_whnf_clo(Clo::of(sort));
        w.spine.is_empty()
            && match self.ctx.read_expr(w.head.e) {
                Sort { level, .. } => self.ctx.is_zero(level),
                _ => false,
            }
    }

    // ---- inference ----

    pub(crate) fn rp_infer(&mut self, c: Clo<'t>, flag: InferFlag) -> ExprPtr<'t> {
        match self.ctx.read_expr(c.e) {
            Var { dbj_idx, .. } => match self.rp_lookup(c.env, dbj_idx) {
                Entry::Neu(fv) => self.rp_fvar_type(fv),
                Entry::Val(e2, env2) => self.rp_infer(Clo { e: e2, env: env2 }, flag),
            },
            Local { binder_type, .. } => binder_type,
            Sort { level, .. } => {
                if flag == Check {
                    self.rp_check_level(level);
                }
                let l2 = self.ctx.succ(level);
                self.ctx.mk_sort(l2)
            }
            Const { name, levels, .. } => self.rp_infer_const(name, levels, flag),
            NatLit { .. } => {
                assert!(self.ctx.export_file.config.nat_extension);
                self.ctx.nat_type().unwrap()
            }
            StringLit { .. } => {
                assert!(self.ctx.export_file.config.string_extension);
                self.ctx.string_type().unwrap()
            }
            Lambda { .. } | Pi { .. } | Let { .. } | App { .. } | Proj { .. } => {
                let key = self.rp_key(c);
                let cached = match flag {
                    Check => self.rp.infer_cache_check.get(&key),
                    InferOnly => self.rp.infer_cache_only.get(&key),
                };
                if let Some(&r) = cached {
                    return r;
                }
                let r = match self.ctx.read_expr(c.e) {
                    Lambda { .. } => self.rp_infer_lambda(c, flag),
                    Pi { .. } => self.rp_infer_pi(c, flag),
                    Let { .. } => self.rp_infer_let(c, flag),
                    App { .. } => {
                        let s = self.rp_mk_sclo(c);
                        self.rp_infer_s(&s, flag)
                    }
                    Proj { ty_name, idx, structure, .. } => {
                        self.rp_infer_proj(ty_name, idx, Clo { e: structure, env: c.env }, flag)
                    }
                    _ => unreachable!(),
                };
                match flag {
                    Check => self.rp.infer_cache_check.insert(key, r),
                    InferOnly => self.rp.infer_cache_only.insert(key, r),
                };
                r
            }
        }
    }

    /// every universe parameter must be bound in the enclosing declaration
    fn rp_check_level(&mut self, l: LevelPtr<'t>) {
        if let Some(di) = self.declar_info {
            assert!(
                self.ctx.all_uparams_defined(l, di.uparams),
                "undefined universe parameter"
            );
        }
    }

    fn rp_infer_const(
        &mut self,
        name: NamePtr<'t>,
        levels: LevelsPtr<'t>,
        flag: InferFlag,
    ) -> ExprPtr<'t> {
        let info = match self.env.get_declar(&name) {
            Some(d) => *d.info(),
            None => panic!("unknown constant in infer_const"),
        };
        assert_eq!(
            self.ctx.read_levels(levels).len(),
            self.ctx.read_levels(info.uparams).len(),
            "incorrect number of universe levels"
        );
        if flag == Check {
            for l in self.ctx.read_levels(levels).iter().copied() {
                self.rp_check_level(l);
            }
        }
        self.ctx.subst_declar_info_levels(info, levels)
    }

    fn rp_ensure_sort(&mut self, ty: ExprPtr<'t>) -> LevelPtr<'t> {
        if let Sort { level, .. } = self.ctx.read_expr(ty) {
            return level;
        }
        let w = self.rp_whnf_clo(Clo::of(ty));
        if w.spine.is_empty() {
            if let Sort { level, .. } = self.ctx.read_expr(w.head.e) {
                return level;
            }
        }
        panic!("type expected")
    }

    fn rp_infer_lambda(&mut self, c: Clo<'t>, flag: InferFlag) -> ExprPtr<'t> {
        let Lambda { binder_name, binder_style, binder_type, body, .. } =
            self.ctx.read_expr(c.e)
        else {
            unreachable!()
        };
        let d = self.rp_reify(Clo { e: binder_type, env: c.env });
        if flag == Check {
            let dty = self.rp_infer(Clo::of(d), flag);
            self.rp_ensure_sort(dty);
        }
        let fv = self.rp_fresh_fvar(d);
        let env2 = self.rp_push_entry(c.env, Entry::Neu(fv));
        let bt = self.rp_infer(Clo { e: body, env: env2 }, flag);
        let bt = self.rp_cheap_beta_reduce(bt);
        let bt_abs = self.rp_abstract1(bt, fv, 0);
        self.ctx.mk_pi(binder_name, binder_style, d, bt_abs)
    }

    fn rp_infer_pi(&mut self, c: Clo<'t>, flag: InferFlag) -> ExprPtr<'t> {
        let Pi { binder_type, body, .. } = self.ctx.read_expr(c.e) else {
            unreachable!()
        };
        let d = self.rp_reify(Clo { e: binder_type, env: c.env });
        let dty = self.rp_infer(Clo::of(d), flag);
        let u = self.rp_ensure_sort(dty);
        let fv = self.rp_fresh_fvar(d);
        let env2 = self.rp_push_entry(c.env, Entry::Neu(fv));
        let bt = self.rp_infer(Clo { e: body, env: env2 }, flag);
        let s = self.rp_ensure_sort(bt);
        // mkLevelIMax': imax with immediate simplifications
        let lvl = self.ctx.imax(u, s);
        let lvl = self.ctx.simplify(lvl);
        self.ctx.mk_sort(lvl)
    }

    /// Abstract the single fvar `fv` in `e` to `Var(depth)`. Memoized.
    fn rp_abstract1(&mut self, e: ExprPtr<'t>, fv: ExprPtr<'t>, depth: u16) -> ExprPtr<'t> {
        if !self.ctx.has_fvars(e) {
            return e;
        }
        let key = (e, fv, depth);
        if let Some(&r) = self.rp.abs_cache.get(&key) {
            return r;
        }
        let r = match self.ctx.read_expr(e) {
            Local { .. } => {
                if e == fv {
                    self.ctx.mk_var(depth)
                } else {
                    e
                }
            }
            App { fun, arg, .. } => {
                let f2 = self.rp_abstract1(fun, fv, depth);
                let x2 = self.rp_abstract1(arg, fv, depth);
                self.ctx.mk_app(f2, x2)
            }
            Lambda { binder_name, binder_style, binder_type, body, .. } => {
                let d2 = self.rp_abstract1(binder_type, fv, depth);
                let b2 = self.rp_abstract1(body, fv, depth + 1);
                self.ctx.mk_lambda(binder_name, binder_style, d2, b2)
            }
            Pi { binder_name, binder_style, binder_type, body, .. } => {
                let d2 = self.rp_abstract1(binder_type, fv, depth);
                let b2 = self.rp_abstract1(body, fv, depth + 1);
                self.ctx.mk_pi(binder_name, binder_style, d2, b2)
            }
            Let { binder_name, binder_type, val, body, nondep, .. } => {
                let t2 = self.rp_abstract1(binder_type, fv, depth);
                let v2 = self.rp_abstract1(val, fv, depth);
                let b2 = self.rp_abstract1(body, fv, depth + 1);
                self.ctx.mk_let(binder_name, t2, v2, b2, nondep)
            }
            Proj { ty_name, idx, structure, .. } => {
                let x2 = self.rp_abstract1(structure, fv, depth);
                self.ctx.mk_proj(ty_name, idx, x2)
            }
            _ => e,
        };
        self.rp.abs_cache.insert(key, r);
        r
    }

    fn rp_infer_let(&mut self, c: Clo<'t>, flag: InferFlag) -> ExprPtr<'t> {
        let Let { binder_type, val, body, .. } = self.ctx.read_expr(c.e) else {
            unreachable!()
        };
        let t = self.rp_reify(Clo { e: binder_type, env: c.env });
        if flag == Check {
            let tty = self.rp_infer(Clo::of(t), flag);
            self.rp_ensure_sort(tty);
            let vty = self.rp_infer(Clo { e: val, env: c.env }, flag);
            assert!(
                self.rp_is_def_eq(Clo::of(vty), Clo::of(t)),
                "let type mismatch"
            );
        }
        let env2 = self.rp_push_entry(c.env, Entry::Val(val, c.env));
        self.rp_infer(Clo { e: body, env: env2 }, flag)
    }

    pub(crate) fn rp_infer_s(&mut self, s: &SClo<'t>, flag: InferFlag) -> ExprPtr<'t> {
        if s.spine.is_empty() {
            return self.rp_infer(s.head, flag);
        }
        let key = self.rp_sclo_key(s);
        if let Some(&r) = self.rp.infer_s_cache.get(&key) {
            return r;
        }
        let mut f_ty: Clo<'t> = Clo::of(self.rp_infer(s.head, flag));
        for &arg in s.spine.iter() {
            let fw = if matches!(self.ctx.read_expr(f_ty.e), Pi { .. }) {
                SClo { head: f_ty, spine: Vec::new() }
            } else {
                self.rp_whnf_clo(f_ty)
            };
            let Pi { binder_type: dom, body, .. } = self.ctx.read_expr(fw.head.e) else {
                panic!("function expected");
            };
            if !fw.spine.is_empty() {
                panic!("function expected");
            }
            if flag == Check {
                let a_ty = self.rp_infer(arg, flag);
                assert!(
                    self.rp_is_def_eq(Clo { e: dom, env: fw.head.env }, Clo::of(a_ty)),
                    "application type mismatch"
                );
            }
            let env2 = self.rp_push_entry(fw.head.env, Entry::Val(arg.e, arg.env));
            f_ty = Clo { e: body, env: env2 };
        }
        let r = self.rp_reify(f_ty);
        self.rp.infer_s_cache.insert(key, r);
        r
    }

    fn rp_infer_proj(
        &mut self,
        type_name: NamePtr<'t>,
        idx: usize,
        strukt: Clo<'t>,
        flag: InferFlag,
    ) -> ExprPtr<'t> {
        let s_ty = self.rp_infer(strukt, flag);
        let st_w = self.rp_whnf_clo(Clo::of(s_ty));
        let (i_name, i_levels) = match self.ctx.read_expr(st_w.head.e) {
            Const { name, levels, .. } => (name, levels),
            _ => panic!("invalid projection"),
        };
        assert!(i_name == type_name, "invalid projection");
        let (num_params, num_indices, ctors) = match self.env.get_inductive(&i_name) {
            Some(ind) => (ind.num_params, ind.num_indices, ind.all_ctor_names.clone()),
            None => panic!("invalid projection"),
        };
        assert!(ctors.len() == 1, "invalid projection");
        assert!(
            st_w.spine.len() == (num_params + num_indices) as usize,
            "invalid projection"
        );
        let ctor_info = *self
            .env
            .get_declar(&ctors[0])
            .expect("invalid projection")
            .info();
        let c_ty = self.ctx.subst_declar_info_levels(ctor_info, i_levels);
        let mut r = Clo::of(c_ty);
        for pi in 0..num_params as usize {
            let rw = self.rp_whnf_clo(r);
            let Pi { body, .. } = self.ctx.read_expr(rw.head.e) else {
                panic!("invalid projection");
            };
            assert!(rw.spine.is_empty(), "invalid projection");
            let p = st_w.spine[pi];
            let env2 = self.rp_push_entry(rw.head.env, Entry::Val(p.e, p.env));
            r = Clo { e: body, env: env2 };
        }
        let is_prop_ty = self.rp_is_prop(s_ty);
        for fi in 0..idx {
            let rw = self.rp_whnf_clo(r);
            let Pi { binder_type: dom, body, .. } = self.ctx.read_expr(rw.head.e) else {
                panic!("invalid projection");
            };
            assert!(rw.spine.is_empty(), "invalid projection");
            if self.lbr(body) > 0 && is_prop_ty {
                let d = self.rp_reify(Clo { e: dom, env: rw.head.env });
                assert!(self.rp_is_prop(d), "invalid projection");
            }
            let bv = self.ctx.mk_var(0);
            let proj = self.ctx.mk_proj(i_name, fi, bv);
            let senv = self.rp_push_entry(ENV_NIL, Entry::Val(strukt.e, strukt.env));
            let env2 = self.rp_push_entry(rw.head.env, Entry::Val(proj, senv));
            r = Clo { e: body, env: env2 };
        }
        let rw = self.rp_whnf_clo(r);
        let Pi { binder_type: dom, .. } = self.ctx.read_expr(rw.head.e) else {
            panic!("invalid projection");
        };
        assert!(rw.spine.is_empty(), "invalid projection");
        if is_prop_ty {
            let d = self.rp_reify(Clo { e: dom, env: rw.head.env });
            assert!(self.rp_is_prop(d), "invalid projection");
        }
        self.rp_reify(Clo { e: dom, env: rw.head.env })
    }

    fn rp_cheap_beta_reduce(&mut self, e: ExprPtr<'t>) -> ExprPtr<'t> {
        if !matches!(self.ctx.read_expr(e), App { .. }) {
            return e;
        }
        let s = self.rp_mk_sclo(Clo::of(e));
        let mut fun = s.head.e;
        let mut i = 0usize;
        while i < s.spine.len() {
            if let Lambda { body, .. } = self.ctx.read_expr(fun) {
                fun = body;
                i += 1;
            } else {
                break;
            }
        }
        if i == 0 {
            return e;
        }
        if self.lbr(fun) == 0 {
            return self
                .ctx
                .foldl_apps(fun, s.spine[i..].iter().map(|c| c.e).collect::<Vec<_>>().into_iter());
        }
        if let Var { dbj_idx: n, .. } = self.ctx.read_expr(fun) {
            if (n as usize) < i {
                let head = s.spine[i - 1 - n as usize].e;
                return self.ctx.foldl_apps(
                    head,
                    s.spine[i..].iter().map(|c| c.e).collect::<Vec<_>>().into_iter(),
                );
            }
        }
        e
    }

    // ---- adapters for nanoda entry points ----

    pub(crate) fn rp_sclo_to_expr(&mut self, s: &SClo<'t>) -> ExprPtr<'t> {
        let mut out = self.rp_reify(s.head);
        for &c in s.spine.iter() {
            let a = self.rp_reify(c);
            out = self.ctx.mk_app(out, a);
        }
        out
    }
}
