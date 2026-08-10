//! Evaluation and reduction of values.
//!
//! `nb_eval` reads an expression under an environment and produces a value.
//! It never rebuilds an expression: entering a binder records the argument in
//! the environment, and an argument is recorded unevaluated so that one that
//! is never looked at is never evaluated. Applying a value that is stuck
//! extends its spine rather than reconstructing the application.
//!
//! `nb_whnf` drives a value to weak head normal form by unfolding constants
//! and firing recursors. Both the result of unfolding a constant and the
//! result of firing a recursor are recorded on the value they came from, so a
//! reduction sequence that is reached twice is walked once.

use crate::env::{Declar, RecursorData};
use crate::expr::Expr;
use crate::nbe::{
    ConstKind, Elim, RigidHead, SpineId, ValId, VEnvId, Value, SPINE_EMPTY, VENV_NIL,
};
use crate::tc::TypeChecker;
use crate::util::{
    nat_div, nat_gcd, nat_land, nat_lor, nat_mod, nat_shl, nat_shr, nat_sub, nat_xor, BigUintPtr,
    ExprPtr, LevelPtr, LevelsPtr, NamePtr, StringPtr,
};
use num_bigint::BigUint;
use num_traits::{pow::Pow, ToPrimitive, Zero};
use Expr::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum NatBinOp {
    Add,
    Sub,
    Mul,
    Pow,
    Gcd,
    Mod,
    Div,
    Beq,
    Ble,
    LAnd,
    LOr,
    XOr,
    Shl,
    Shr,
}

impl<'x, 't: 'x, 'p: 't> TypeChecker<'x, 't, 'p> {
    // ---- evaluation ----

    /// The value of `e` under `env`. There is no memo over this: a closed
    /// expression evaluates under the empty environment and interns to one
    /// value, a delayed argument evaluates through its thunk's forced cell,
    /// and an unfolded constant through its `Unfold` node, so the sharing an
    /// evaluation memo would buy already lives in the values themselves.
    pub(crate) fn nb_eval(&mut self, depth: u32, env: VEnvId, e: ExprPtr<'t>) -> ValId {
        let env = if self.ctx.num_loose_bvars(e) == 0 { VENV_NIL } else { env };
        self.nb_eval_go(depth, env, e)
    }

    fn nb_eval_go(&mut self, depth: u32, env: VEnvId, e: ExprPtr<'t>) -> ValId {
        match self.ctx.read_expr(e) {
            Var { dbj_idx, .. } => {
                let entry = self.lookup(env, dbj_idx);
                let v = self.entry_val(entry);
                self.nb_force(depth, v)
            }
            Sort { level, .. } => {
                let level = self.ctx.simplify(level);
                self.ctx.nb.mk_sort(level)
            }
            NatLit { ptr, .. } => self.ctx.nb.mk_nat(ptr),
            StringLit { ptr, .. } => self.ctx.nb.mk_str(ptr),
            Local { .. } => self.nb_local(e),
            Const { name, levels, .. } => self.nb_const(name, levels),
            App { .. } => {
                // The whole application chain at once: a lambda head consumes
                // arguments through the environment, and the first stuck head
                // takes every remaining argument onto its spine as one value.
                let mut args: smallvec::SmallVec<[ExprPtr<'t>; 8]> = smallvec::SmallVec::new();
                let mut cursor = e;
                while let App { fun, arg, .. } = self.ctx.read_expr(cursor) {
                    args.push(arg);
                    cursor = fun;
                }
                let mut f = self.nb_eval(depth, env, cursor);
                let mut i = args.len();
                while i > 0 {
                    let batchable = match self.ctx.nb.get(f) {
                        Value::Rigid { head, .. } => !matches!(
                            head,
                            RigidHead::Const(ConstKind::Ctor, name, _)
                                if self.nat_ext()
                                    && Some(name) == self.ctx.export_file.name_cache.nat_succ
                        ),
                        Value::Unfold { name, .. } => {
                            !(self.nat_ext() && self.nb_is_nat_prim(name))
                        }
                        _ => false,
                    };
                    if batchable {
                        let (head_rigid, mut spine) = match self.ctx.nb.get(f) {
                            Value::Rigid { head, spine } => (Ok(head), spine),
                            Value::Unfold { name, levels, spine, .. } => {
                                (Err((name, levels)), spine)
                            }
                            _ => unreachable!(),
                        };
                        while i > 0 {
                            i -= 1;
                            let a = self.nb_delay(depth, env, args[i]);
                            spine = self.ctx.nb.spine_snoc(spine, Elim::App(a));
                        }
                        f = match head_rigid {
                            Ok(head) => self.ctx.nb.mk_rigid(head, spine),
                            Err((name, levels)) => self.ctx.nb.mk_unfold(name, levels, spine),
                        };
                        break;
                    }
                    i -= 1;
                    let a = self.nb_delay(depth, env, args[i]);
                    f = self.nb_apply(depth, f, a);
                }
                f
            }
            Lambda { binder_name, binder_style, binder_type, body, .. } => {
                self.ctx.nb.mk_lam(binder_name, binder_style, binder_type, env, body)
            }
            Pi { binder_name, binder_style, binder_type, body, .. } => {
                let domain = self.nb_delay(depth, env, binder_type);
                self.ctx.nb.mk_pi(binder_name, binder_style, domain, env, body)
            }
            Let { .. } => {
                let mut env = env;
                let mut cursor = e;
                while let Let { val, body, .. } = self.ctx.read_expr(cursor) {
                    let v = self.nb_eval(depth, env, val);
                    env = self.push_entry_v(env, v);
                    cursor = body;
                }
                self.nb_eval(depth, env, cursor)
            }
            Proj { ty_name, idx, structure, .. } => {
                let s = self.nb_eval(depth, env, structure);
                self.nb_proj(depth, ty_name, idx, s)
            }
        }
    }

    /// Record a subterm without evaluating it. Something already in normal
    /// form is evaluated straight away, since a thunk over it would cost more
    /// than the value.
    fn nb_delay(&mut self, depth: u32, env: VEnvId, e: ExprPtr<'t>) -> ValId {
        match self.ctx.read_expr(e) {
            Var { .. } | Sort { .. } | NatLit { .. } | StringLit { .. } | Local { .. }
            | Const { .. } => self.nb_eval(depth, env, e),
            _ => {
                let env = if self.ctx.num_loose_bvars(e) == 0 { VENV_NIL } else { env };
                self.ctx.nb.mk_thunk(env, e)
            }
        }
    }

    /// The value an environment entry stands for: an evaluated entry is
    /// itself, a delayed one becomes a thunk, an opened binder its neutral.
    pub(crate) fn entry_val(&mut self, entry: crate::closure::Entry<'t>) -> crate::nbe::ValId {
        match entry {
            crate::closure::Entry::V(v) => v,
            crate::closure::Entry::Val(e, env) => {
                let env = if self.lbr(e) == 0 { crate::nbe::VENV_NIL } else { env };
                self.ctx.nb.mk_thunk(env, e)
            }
            crate::closure::Entry::Neu(fv) => self.nb_local(fv),
        }
    }

    pub(crate) fn nb_local(&mut self, e: ExprPtr<'t>) -> ValId {
        if let Some(&v) = self.ctx.nb.local_cache.get(&e) {
            return v;
        }
        let v = self.ctx.nb.mk_rigid(RigidHead::Local(e), SPINE_EMPTY);
        self.ctx.nb.local_cache.insert(e, v);
        v
    }

    /// The value denoting a constant: one with a body stays folded, anything
    /// else is a neutral carrying the kind that decides how it reduces.
    fn nb_const(&mut self, name: NamePtr<'t>, levels: LevelsPtr<'t>) -> ValId {
        if let Some(&v) = self.ctx.nb.const_val_cache.get(&(name, levels)) {
            return v;
        }
        let v = match self.env.get_declar(&name) {
            Some(Declar::Definition { .. } | Declar::Theorem { .. }) => {
                self.ctx.nb.mk_unfold(name, levels, SPINE_EMPTY)
            }
            Some(d) => {
                let kind = match d {
                    Declar::Constructor(..) => ConstKind::Ctor,
                    Declar::Recursor(..) => ConstKind::Recursor,
                    Declar::Quot { .. } => ConstKind::QuotConst,
                    Declar::Inductive(..) => ConstKind::Inductive,
                    _ => ConstKind::Axiom,
                };
                self.ctx.nb.mk_rigid(RigidHead::Const(kind, name, levels), SPINE_EMPTY)
            }
            None => self.ctx.nb.mk_rigid(RigidHead::Const(ConstKind::Axiom, name, levels), SPINE_EMPTY),
        };
        self.ctx.nb.const_val_cache.insert((name, levels), v);
        v
    }

    /// Force a thunk, remembering the result in it.
    pub(crate) fn nb_force(&mut self, depth: u32, v: ValId) -> ValId {
        match self.ctx.nb.get(v) {
            Value::Thunk { env, expr, forced } => {
                if let Some(f) = forced {
                    return f;
                }
                let f = self.nb_eval(depth, env, expr);
                let f = self.nb_force(depth, f);
                self.ctx.nb.set_forced(v, f);
                f
            }
            _ => v,
        }
    }

    /// Apply a value to an argument.
    pub(crate) fn nb_apply(&mut self, depth: u32, f: ValId, a: ValId) -> ValId {
        match self.ctx.nb.get(f) {
            Value::Lam { env, body, .. } => {
                let env2 = self.push_entry_v(env, a);
                self.nb_eval(depth, env2, body)
            }
            Value::Rigid { head, spine } => {
                let spine = self.ctx.nb.spine_snoc(spine, Elim::App(a));
                // `Nat.succ` of a literal is a literal, so a chain of them
                // does not grow a unary spine.
                if let RigidHead::Const(ConstKind::Ctor, name, _) = head {
                    if self.nat_ext() && Some(name) == self.ctx.export_file.name_cache.nat_succ {
                        if let Some(n) = self.nb_bignum(depth, a, false) {
                            if let Some(p) = self.ctx.alloc_bignum(n + 1u8) {
                                return self.ctx.nb.mk_nat(p);
                            }
                        }
                    }
                }
                self.ctx.nb.mk_rigid(head, spine)
            }
            Value::Unfold { name, levels, spine, .. } => {
                let spine = self.ctx.nb.spine_snoc(spine, Elim::App(a));
                // A `Nat` primitive whose arguments are already literals is
                // answered here, so the recursive definition never unfolds.
                if self.nat_ext() && self.nb_is_nat_prim(name) {
                    if let Some(args) = self.ctx.nb.spine_args(spine) {
                        if let Some(r) = self.nb_nat_red(depth, name, &args, false) {
                            return r;
                        }
                    }
                }
                self.ctx.nb.mk_unfold(name, levels, spine)
            }
            Value::Thunk { .. } => {
                let f = self.nb_force(depth, f);
                self.nb_apply(depth, f, a)
            }
            _ => panic!("nb_apply: not a function"),
        }
    }

    /// Project a field out of a value.
    pub(crate) fn nb_proj(
        &mut self,
        depth: u32,
        ty_name: NamePtr<'t>,
        idx: usize,
        s: ValId,
    ) -> ValId {
        let s = self.nb_whnf(depth, s);
        match self.ctx.nb.get(s) {
            Value::Rigid { head: RigidHead::Const(ConstKind::Ctor, ctor, _), spine } => {
                if let Some(cd) = self.env.get_constructor(&ctor) {
                    if cd.inductive_name == ty_name {
                        let np = usize::from(cd.num_params);
                        if let Some(Elim::App(field)) = self.ctx.nb.spine_get(spine, np + idx) {
                            return self.nb_force(depth, field);
                        }
                    }
                }
                self.nb_proj_stuck(ty_name, idx, s)
            }
            Value::NatLit { ptr } => {
                let c = self.nb_nat_to_ctor(depth, ptr).expect("nb_proj: nat literal");
                self.nb_proj(depth, ty_name, idx, c)
            }
            Value::StrLit { ptr } => {
                let c = self.nb_str_to_ctor(depth, ptr).expect("nb_proj: string literal");
                self.nb_proj(depth, ty_name, idx, c)
            }
            Value::Rigid { .. } | Value::Unfold { .. } => self.nb_proj_stuck(ty_name, idx, s),
            other => panic!(
                "nb_proj: not a structure: {} .{} of {}",
                format!("{:?}", self.ctx.debug_print(ty_name)),
                idx,
                match other {
                    Value::Lam { .. } => "lambda",
                    Value::Pi { .. } => "pi",
                    Value::Sort { .. } => "sort",
                    _ => "?",
                }
            ),
        }
    }

    fn nb_proj_stuck(&mut self, ty_name: NamePtr<'t>, idx: usize, s: ValId) -> ValId {
        match self.ctx.nb.get(s) {
            Value::Rigid { head, spine } => {
                let spine = self.ctx.nb.spine_snoc(spine, Elim::Proj { ty_name, idx });
                self.ctx.nb.mk_rigid(head, spine)
            }
            Value::Unfold { name, levels, spine, .. } => {
                let spine = self.ctx.nb.spine_snoc(spine, Elim::Proj { ty_name, idx });
                self.ctx.nb.mk_unfold(name, levels, spine)
            }
            _ => unreachable!("nb_proj_stuck: not neutral"),
        }
    }

    /// The domain of a lambda or pi, evaluated once.
    pub(crate) fn nb_lam_domain(&mut self, depth: u32, v: ValId) -> ValId {
        match self.ctx.nb.get(v) {
            Value::Lam { binder_type, domain, env, .. } => {
                if let Some(d) = domain {
                    return d;
                }
                let d = self.nb_eval(depth, env, binder_type);
                self.ctx.nb.set_domain(v, d);
                d
            }
            Value::Pi { domain, .. } => self.nb_force(depth, domain),
            _ => panic!("nb_lam_domain: not a binder"),
        }
    }

    /// Instantiate a binder's body at `a`.
    pub(crate) fn nb_open(&mut self, depth: u32, binder: ValId, a: ValId) -> ValId {
        let (env, body) = match self.ctx.nb.get(binder) {
            Value::Lam { env, body, .. } | Value::Pi { env, body, .. } => (env, body),
            _ => panic!("nb_open: not a binder"),
        };
        let env2 = self.push_entry_v(env, a);
        self.nb_eval(depth, env2, body)
    }

    // ---- reduction ----

    /// Weak head normal form: unfold constants and fire recursors until the
    /// head is stuck.
    pub(crate) fn nb_whnf(&mut self, depth: u32, v: ValId) -> ValId {
        stacker::maybe_grow(256 * 1024, 16 * 1024 * 1024, || {
            let mut cur = v;
            loop {
                cur = self.nb_force(depth, cur);
                match self.ctx.nb.get(cur) {
                    Value::Unfold { .. } => {
                        let next = self.nb_unfold(depth, cur);
                        if next == cur {
                            return cur;
                        }
                        cur = next;
                    }
                    Value::Rigid { head: RigidHead::Const(k, ..), .. }
                        if matches!(k, ConstKind::Recursor | ConstKind::QuotConst) =>
                    {
                        match self.nb_iota(depth, cur) {
                            Some(next) => cur = next,
                            None => return cur,
                        }
                    }
                    _ => return cur,
                }
            }
        })
    }

    /// Unfold a folded constant application by evaluating the definition's
    /// body and replaying the spine on it. The result is recorded on the
    /// value, so the replay happens once.
    pub(crate) fn nb_unfold(&mut self, depth: u32, v: ValId) -> ValId {
        self.nb_unfold_go(depth, v, false)
    }

    /// Unfold even a `Nat` primitive that would rather stay folded. Used
    /// where conversion has nothing else left to try.
    pub(crate) fn nb_unfold_demand(&mut self, depth: u32, v: ValId) -> ValId {
        let force = self.ctx.nb.probe_depth == 0;
        self.nb_unfold_go(depth, v, force)
    }

    fn nb_unfold_go(&mut self, depth: u32, v: ValId, force: bool) -> ValId {
        let Value::Unfold { name, levels, spine, forced } = self.ctx.nb.get(v) else {
            return v;
        };
        if let Some(f) = forced {
            return f;
        }
        if self.nat_ext() && self.nb_is_nat_prim(name) {
            if let Some(args) = self.ctx.nb.spine_args(spine) {
                if let Some(r) = self.nb_nat_red(depth, name, &args, true) {
                    self.ctx.nb.set_forced(v, r);
                    return r;
                }
                // `Nat.add`, and its siblings, recurse on their second
                // argument. Unfolding one whose second argument is a large
                // literal walks that literal in unary, so the application
                // stays folded until something demands otherwise.
                if !force && self.nb_nat_defer(depth, name, &args) {
                    return v;
                }
            }
        }
        let Some(head) = self.nb_unfold_const(name, levels) else {
            self.ctx.nb.set_forced(v, v);
            return v;
        };
        let mut cur = head;
        for elim in self.ctx.nb.spine_to_vec(spine) {
            cur = match elim {
                Elim::App(a) => self.nb_apply(depth, cur, a),
                Elim::Proj { ty_name, idx } => self.nb_proj(depth, ty_name, idx, cur),
            };
        }
        self.ctx.nb.set_forced(v, cur);
        cur
    }

    /// The body of a constant, with its level parameters replaced, evaluated
    /// once per constant and shared by every occurrence.
    pub(crate) fn nb_unfold_const(
        &mut self,
        name: NamePtr<'t>,
        levels: LevelsPtr<'t>,
    ) -> Option<ValId> {
        if let Some(&v) = self.ctx.nb.unfold_cache.get(&(name, levels)) {
            return v;
        }
        let r = (|| {
            let (uparams, val) = self.env.get_declar_val(&name)?;
            if self.ctx.read_levels(levels).len() != self.ctx.read_levels(uparams).len() {
                return None;
            }
            // The instantiated body is shared across declarations through
            // the same cache the closure machine reads, under its lifetime
            // discipline: closed, environment-free, constants only.
            let val = if self.ctx.read_levels(levels).is_empty() {
                val
            } else if !self.env.has_temp_ext() {
                let key = self.ctx.mk_const(name, levels);
                if let Some(&r) = self.ctx.rp.g_unfold.get(&key) {
                    r
                } else {
                    let r = self.ctx.subst_expr_levels(val, uparams, levels);
                    self.ctx.rp.g_unfold.insert(key, r);
                    r
                }
            } else {
                self.ctx.subst_expr_levels(val, uparams, levels)
            };
            Some(self.nb_eval(0, VENV_NIL, val))
        })();
        self.ctx.nb.unfold_cache.insert((name, levels), r);
        r
    }

    /// Fire a recursor or a quotient eliminator standing at the head of `v`,
    /// or report it stuck. Both answers are recorded on `v`.
    pub(crate) fn nb_iota(&mut self, depth: u32, v: ValId) -> Option<ValId> {
        if let Some(&r) = self.ctx.nb.iota_cache.get(&v) {
            return r;
        }
        let r = self.nb_iota_go(depth, v);
        self.ctx.nb.iota_cache.insert(v, r);
        r
    }

    fn nb_iota_go(&mut self, depth: u32, v: ValId) -> Option<ValId> {
        let Value::Rigid { head: RigidHead::Const(kind, name, levels), spine } =
            self.ctx.nb.get(v)
        else {
            return None;
        };
        let args = self.ctx.nb.spine_args(spine)?;
        match kind {
            ConstKind::Recursor => {
                let env = self.env;
                let rec = env.get_recursor(&name)?;
                if self.ctx.read_levels(rec.info.uparams).len()
                    != self.ctx.read_levels(levels).len()
                {
                    return None;
                }
                if args.len() <= rec.major_idx() {
                    return None;
                }
                if let Some(r) = self.nb_k_pre_reduce(depth, rec, levels, &args) {
                    return Some(r);
                }
                let major = self.nb_whnf(depth, args[rec.major_idx()]);
                self.nb_fire_recursor(depth, rec, levels, &args, major)
            }
            ConstKind::QuotConst => {
                let nc = self.ctx.export_file.name_cache;
                let mk_pos = if Some(name) == nc.quot_lift {
                    5usize
                } else if Some(name) == nc.quot_ind {
                    4usize
                } else {
                    return None;
                };
                let major = self.nb_whnf(depth, *args.get(mk_pos)?);
                self.nb_fire_quot(depth, name, &args, major)
            }
            _ => None,
        }
    }

    fn nb_fire_quot(
        &mut self,
        depth: u32,
        name: NamePtr<'t>,
        args: &[ValId],
        major: ValId,
    ) -> Option<ValId> {
        let nc = self.ctx.export_file.name_cache;
        let rest_idx = if Some(name) == nc.quot_lift {
            6usize
        } else if Some(name) == nc.quot_ind {
            5usize
        } else {
            return None;
        };
        let Value::Rigid { head: RigidHead::Const(ConstKind::QuotConst, mk, _), spine } =
            self.ctx.nb.get(major)
        else {
            return None;
        };
        if Some(mk) != nc.quot_mk {
            return None;
        }
        let mk_args = self.ctx.nb.spine_args(spine)?;
        if mk_args.len() != 3 {
            return None;
        }
        let f = *args.get(3)?;
        let mut result = self.nb_apply(depth, f, mk_args[2]);
        for &a in &args[rest_idx..] {
            result = self.nb_apply(depth, result, a);
        }
        Some(result)
    }

    fn nb_fire_recursor(
        &mut self,
        depth: u32,
        rec: &RecursorData<'t>,
        levels: LevelsPtr<'t>,
        args: &[ValId],
        major: ValId,
    ) -> Option<ValId> {
        // `Nat.rec` on a literal steps without expanding the literal into a
        // tower of `Nat.succ`.
        if self.nat_ext()
            && rec.all_inductives.first().copied() == self.ctx.export_file.name_cache.nat
        {
            if let Value::NatLit { ptr } = self.ctx.nb.get(major) {
                return Some(self.nb_nat_rec(depth, args, ptr, rec, levels));
            }
        }
        let major = self
            .nb_major_to_ctor(depth, major)
            .or_else(|| self.nb_k_reduce(depth, major, rec))
            .or_else(|| self.nb_struct_eta_reduce(depth, major, rec))
            .unwrap_or(major);
        let Value::Rigid { head: RigidHead::Const(ConstKind::Ctor, ctor, _), spine } =
            self.ctx.nb.get(major)
        else {
            return None;
        };
        let ctor_args = self.ctx.nb.spine_args(spine)?;
        let rule = rec.rec_rules.iter().find(|r| r.ctor_name == ctor).copied()?;
        let nfields = usize::from(rule.ctor_telescope_size_wo_params);
        let num_extra = ctor_args.len().checked_sub(nfields)?;
        let mut result = match self.ctx.nb.rec_rule_cache.get(&(rule.val, levels)) {
            Some(&v) => v,
            None => {
                let body = self.ctx.subst_expr_levels(rule.val, rec.info.uparams, levels);
                let v = self.nb_eval(0, VENV_NIL, body);
                self.ctx.nb.rec_rule_cache.insert((rule.val, levels), v);
                v
            }
        };
        let nprefix = usize::from(rec.num_params + rec.num_motives + rec.num_minors);
        for &a in &args[..nprefix] {
            result = self.nb_apply(depth, result, a);
        }
        for &a in &ctor_args[num_extra..] {
            result = self.nb_apply(depth, result, a);
        }
        for &a in &args[rec.major_idx() + 1..] {
            result = self.nb_apply(depth, result, a);
        }
        Some(result)
    }

    /// `Nat.rec` at a literal: the zero case, or the successor case at the
    /// predecessor with the recursive call left folded.
    fn nb_nat_rec(
        &mut self,
        depth: u32,
        args: &[ValId],
        n_ptr: BigUintPtr<'t>,
        rec: &RecursorData<'t>,
        levels: LevelsPtr<'t>,
    ) -> ValId {
        let n = self.ctx.read_bignum(n_ptr).expect("nb_nat_rec: literal").clone();
        let nparams = usize::from(rec.num_params);
        let nmotives = usize::from(rec.num_motives);
        let major_idx = rec.major_idx();
        let mut result = if n.is_zero() {
            args[nparams + nmotives]
        } else {
            let pred = self.ctx.alloc_bignum(n - 1u8).expect("nb_nat_rec: predecessor");
            let pred_val = self.ctx.nb.mk_nat(pred);
            let succ_case = self.nb_force(depth, args[nparams + nmotives + 1]);
            let mut ih = self.ctx.nb.mk_rigid(
                RigidHead::Const(ConstKind::Recursor, rec.info.name, levels),
                SPINE_EMPTY,
            );
            for &a in &args[..major_idx] {
                ih = self.nb_apply(depth, ih, a);
            }
            ih = self.nb_apply(depth, ih, pred_val);
            let stepped = self.nb_apply(depth, succ_case, pred_val);
            self.nb_apply(depth, stepped, ih)
        };
        for &a in &args[major_idx + 1..] {
            result = self.nb_apply(depth, result, a);
        }
        result
    }

    fn nb_major_to_ctor(&mut self, depth: u32, major: ValId) -> Option<ValId> {
        match self.ctx.nb.get(major) {
            Value::NatLit { ptr } => self.nb_nat_to_ctor(depth, ptr),
            Value::StrLit { ptr } => self.nb_str_to_ctor(depth, ptr),
            _ => None,
        }
    }

    fn nb_nat_to_ctor(&mut self, depth: u32, n: BigUintPtr<'t>) -> Option<ValId> {
        if !self.nat_ext() {
            return None;
        }
        let _ = depth;
        let nv = self.ctx.read_bignum(n)?.clone();
        let levels = self.ctx.alloc_levels_slice(&[]);
        if nv.is_zero() {
            let zero = self.ctx.export_file.name_cache.nat_zero?;
            Some(self.ctx.nb.mk_rigid(RigidHead::Const(ConstKind::Ctor, zero, levels), SPINE_EMPTY))
        } else {
            let pred = self.ctx.alloc_bignum(nv - 1u8)?;
            let pred_v = self.ctx.nb.mk_nat(pred);
            let succ = self.ctx.export_file.name_cache.nat_succ?;
            let spine = self.ctx.nb.spine_snoc(SPINE_EMPTY, Elim::App(pred_v));
            Some(self.ctx.nb.mk_rigid(RigidHead::Const(ConstKind::Ctor, succ, levels), spine))
        }
    }

    fn nb_str_to_ctor(&mut self, depth: u32, s: StringPtr<'t>) -> Option<ValId> {
        let e = self.ctx.str_lit_to_constructor(s)?;
        let v = self.nb_eval(depth, VENV_NIL, e);
        Some(self.nb_whnf(depth, v))
    }

    /// The K rule: a proof of an inductive proposition with one nullary
    /// constructor is that constructor, so the recursor can fire on it.
    fn nb_k_pre_reduce(
        &mut self,
        depth: u32,
        rec: &RecursorData<'t>,
        levels: LevelsPtr<'t>,
        args: &[ValId],
    ) -> Option<ValId> {
        if !rec.is_k {
            return None;
        }
        let raw = self.nb_force(depth, args[rec.major_idx()]);
        let kctor = self.nb_k_reduce(depth, raw, rec)?;
        self.nb_fire_recursor(depth, rec, levels, args, kctor)
    }

    fn nb_k_reduce(&mut self, depth: u32, major: ValId, rec: &RecursorData<'t>) -> Option<ValId> {
        if !rec.is_k {
            return None;
        }
        if !matches!(self.ctx.nb.get(major), Value::Rigid { .. } | Value::Unfold { .. }) {
            return None;
        }
        let major_ty = self.nb_type(depth, major);
        let major_ty = self.nb_whnf(depth, major_ty);
        let (ty_name, ty_levels, ty_args) = self.nb_as_inductive(major_ty)?;
        let rec_induct = self.ctx.get_major_induct(rec)?;
        if ty_name != rec_induct {
            return None;
        }
        let ind = self.env.get_inductive(&ty_name)?;
        let ctor_name = *ind.all_ctor_names.first()?;
        let np = usize::from(rec.num_params);
        let ctor_self = rec
            .rec_rules
            .iter()
            .find(|r| r.ctor_name == ctor_name)
            .map(|r| usize::from(r.ctor_telescope_size_wo_params))
            .unwrap_or(0);
        let take = (np + ctor_self).min(ty_args.len());
        let mut new_ctor = self
            .ctx
            .nb
            .mk_rigid(RigidHead::Const(ConstKind::Ctor, ctor_name, ty_levels), SPINE_EMPTY);
        for &a in ty_args.iter().take(take) {
            new_ctor = self.nb_apply(depth, new_ctor, a);
        }
        let new_ty = self.nb_type(depth, new_ctor);
        if !self.nb_conv(depth, major_ty, new_ty) {
            return None;
        }
        Some(new_ctor)
    }

    /// A value of a structure type is the constructor applied to its
    /// projections, so a recursor over a structure fires on anything.
    fn nb_struct_eta_reduce(
        &mut self,
        depth: u32,
        major: ValId,
        rec: &RecursorData<'t>,
    ) -> Option<ValId> {
        if !matches!(self.ctx.nb.get(major), Value::Rigid { .. } | Value::Unfold { .. }) {
            return None;
        }
        let rec_induct = self.ctx.get_major_induct(rec)?;
        if !self.env.can_be_struct(&rec_induct) {
            return None;
        }
        if let Some(&r) = self.ctx.nb.struct_eta_cache.get(&(major, rec_induct)) {
            return r;
        }
        let np = usize::from(rec.num_params);
        let r = self.nb_struct_eta_go(depth, major, rec_induct, np);
        self.ctx.nb.struct_eta_cache.insert((major, rec_induct), r);
        r
    }

    fn nb_struct_eta_go(
        &mut self,
        depth: u32,
        major: ValId,
        rec_induct: NamePtr<'t>,
        np: usize,
    ) -> Option<ValId> {
        let major_ty = self.nb_type(depth, major);
        let major_ty = self.nb_whnf(depth, major_ty);
        let (ty_name, ty_levels, ty_args) = self.nb_as_inductive(major_ty)?;
        if ty_name != rec_induct {
            return None;
        }
        // A structure whose universe an instantiation may send to zero is
        // left alone: expanding a proof into its fields would equate proofs
        // that proof irrelevance already equates on other grounds.
        if self.nb_may_be_prop(depth, major_ty) {
            return None;
        }
        let ind = self.env.get_inductive(&ty_name)?;
        let ctor_name = *ind.all_ctor_names.first()?;
        let num_fields = usize::from(self.env.get_constructor(&ctor_name)?.num_fields);
        let mut new_ctor = self
            .ctx
            .nb
            .mk_rigid(RigidHead::Const(ConstKind::Ctor, ctor_name, ty_levels), SPINE_EMPTY);
        for &a in ty_args.iter().take(np) {
            new_ctor = self.nb_apply(depth, new_ctor, a);
        }
        for i in 0..num_fields {
            let proj = self.nb_proj(depth, ty_name, i, major);
            new_ctor = self.nb_apply(depth, new_ctor, proj);
        }
        Some(new_ctor)
    }

    pub(crate) fn nb_as_inductive(
        &mut self,
        v: ValId,
    ) -> Option<(NamePtr<'t>, LevelsPtr<'t>, Vec<ValId>)> {
        match self.ctx.nb.get(v) {
            Value::Rigid { head: RigidHead::Const(ConstKind::Inductive, n, ls), spine } => {
                let args = self.ctx.nb.spine_args(spine)?;
                Some((n, ls, args))
            }
            _ => None,
        }
    }

    // ---- the Nat extension ----

    #[inline]
    fn nat_ext(&self) -> bool { self.ctx.export_file.config.nat_extension }

    fn nb_is_nat_prim(&self, name: NamePtr<'t>) -> bool {
        let nc = &self.ctx.export_file.name_cache;
        let n = Some(name);
        n == nc.nat_succ
            || n == nc.nat_add
            || n == nc.nat_sub
            || n == nc.nat_mul
            || n == nc.nat_pow
            || n == nc.nat_mod
            || n == nc.nat_div
            || n == nc.nat_beq
            || n == nc.nat_ble
            || n == nc.nat_land
            || n == nc.nat_lor
            || n == nc.nat_xor
            || n == nc.nat_gcd
            || n == nc.nat_shl
            || n == nc.nat_shr
    }

    /// Whether unfolding this application would walk a literal in unary.
    fn nb_nat_defer(&mut self, depth: u32, name: NamePtr<'t>, args: &[ValId]) -> bool {
        let nc = &self.ctx.export_file.name_cache;
        let n = Some(name);
        let structural_on_second =
            n == nc.nat_add || n == nc.nat_sub || n == nc.nat_mul || n == nc.nat_pow;
        if !structural_on_second || args.len() != 2 {
            return false;
        }
        let a = self.nb_force(depth, args[1]);
        match self.ctx.nb.get(a) {
            Value::NatLit { ptr } => {
                self.ctx.read_bignum(ptr).map(|n| n.bits() > 8).unwrap_or(false)
            }
            _ => false,
        }
    }

    /// Answer a `Nat` primitive whose arguments are literals. `deep` allows
    /// an argument to be reduced to reach its literal; without it only
    /// arguments that are already literals count.
    fn nb_nat_red(
        &mut self,
        depth: u32,
        name: NamePtr<'t>,
        args: &[ValId],
        deep: bool,
    ) -> Option<ValId> {
        let nc = self.ctx.export_file.name_cache;
        if args.len() == 1 && Some(name) == nc.nat_succ {
            let n = self.nb_bignum(depth, args[0], deep)?;
            return self.nb_mk_nat(n + 1u8);
        }
        if args.len() != 2 {
            return None;
        }
        use NatBinOp::*;
        let n = Some(name);
        let op = if n == nc.nat_add {
            Add
        } else if n == nc.nat_sub {
            Sub
        } else if n == nc.nat_mul {
            Mul
        } else if n == nc.nat_pow {
            Pow
        } else if n == nc.nat_mod {
            Mod
        } else if n == nc.nat_div {
            Div
        } else if n == nc.nat_beq {
            Beq
        } else if n == nc.nat_ble {
            Ble
        } else if n == nc.nat_land {
            LAnd
        } else if n == nc.nat_lor {
            LOr
        } else if n == nc.nat_xor {
            XOr
        } else if n == nc.nat_gcd {
            Gcd
        } else if n == nc.nat_shl {
            Shl
        } else if n == nc.nat_shr {
            Shr
        } else {
            return None;
        };
        let y = self.nb_bignum(depth, args[1], deep)?;
        // the exponent and shift bounds are checked before the base is read
        if matches!(op, Pow | Shl) && y > BigUint::from(1u32 << 24) {
            return None;
        }
        let x = self.nb_bignum(depth, args[0], deep)?;
        match op {
            Add => self.nb_mk_nat(x + y),
            Sub => self.nb_mk_nat(nat_sub(x, y)),
            Mul => self.nb_mk_nat(x * y),
            Pow => self.nb_mk_nat(x.pow(y.to_u32()?)),
            Div => self.nb_mk_nat(nat_div(x, y)),
            Mod => self.nb_mk_nat(nat_mod(x, y)),
            Gcd => self.nb_mk_nat(nat_gcd(&x, &y)),
            LAnd => self.nb_mk_nat(nat_land(x, y)),
            LOr => self.nb_mk_nat(nat_lor(x, y)),
            XOr => self.nb_mk_nat(nat_xor(&x, &y)),
            Shl => self.nb_mk_nat(nat_shl(x, y)),
            Shr => self.nb_mk_nat(nat_shr(x, y)),
            Beq => self.nb_mk_bool(x == y),
            Ble => self.nb_mk_bool(x <= y),
        }
    }

    fn nb_mk_nat(&mut self, n: BigUint) -> Option<ValId> {
        let p = self.ctx.alloc_bignum(n)?;
        Some(self.ctx.nb.mk_nat(p))
    }

    fn nb_mk_bool(&mut self, b: bool) -> Option<ValId> {
        let nc = self.ctx.export_file.name_cache;
        let name = if b { nc.bool_true? } else { nc.bool_false? };
        let levels = self.ctx.alloc_levels_slice(&[]);
        Some(self.ctx.nb.mk_rigid(RigidHead::Const(ConstKind::Ctor, name, levels), SPINE_EMPTY))
    }

    /// The natural number a value denotes, counting `Nat.succ` applications
    /// down to a literal or `Nat.zero`.
    pub(crate) fn nb_bignum(&mut self, depth: u32, v: ValId, deep: bool) -> Option<BigUint> {
        let nc = self.ctx.export_file.name_cache;
        let mut succs: u64 = 0;
        let mut cur = self.nb_force(depth, v);
        loop {
            match self.ctx.nb.get(cur) {
                Value::NatLit { ptr } => {
                    return self.ctx.read_bignum(ptr).cloned().map(|n| n + succs);
                }
                Value::Rigid { head: RigidHead::Const(ConstKind::Ctor, name, _), spine } => {
                    if Some(name) == nc.nat_zero && self.ctx.nb.spine_len(spine) == 0 {
                        return Some(BigUint::from(succs));
                    }
                    if Some(name) == nc.nat_succ && self.ctx.nb.spine_len(spine) == 1 {
                        if let Some(Elim::App(a)) = self.ctx.nb.spine_get(spine, 0) {
                            succs += 1;
                            cur = self.nb_force(depth, a);
                            continue;
                        }
                    }
                    return None;
                }
                Value::Unfold { forced: Some(f), .. } if f != cur => {
                    cur = f;
                }
                Value::Unfold { .. }
                | Value::Rigid { head: RigidHead::Const(ConstKind::Recursor, ..), .. }
                | Value::Rigid { head: RigidHead::Const(ConstKind::QuotConst, ..), .. } => {
                    if !deep || self.nb_is_open(depth, cur) {
                        return None;
                    }
                    let f = self.nb_whnf(depth, cur);
                    if f == cur {
                        return None;
                    }
                    cur = f;
                }
                _ => return None,
            }
        }
    }

    /// Whether an opened binder or a free variable occurs in a value. A value
    /// that has none can be reduced without risk of getting stuck on one.
    pub(crate) fn nb_is_open(&mut self, depth: u32, v: ValId) -> bool {
        let v = self.nb_force(depth, v);
        if let Some(&b) = self.ctx.nb.open_cache.get(&v) {
            return b;
        }
        let r = match self.ctx.nb.get(v) {
            Value::Sort { .. } | Value::NatLit { .. } | Value::StrLit { .. } => false,
            Value::Rigid { head: RigidHead::BVar(..) | RigidHead::Local(..), .. } => true,
            Value::Rigid { spine, .. } | Value::Unfold { spine, .. } => {
                let mut found = false;
                for elim in self.ctx.nb.spine_to_vec(spine) {
                    if let Elim::App(a) = elim {
                        if self.nb_is_open(depth, a) {
                            found = true;
                            break;
                        }
                    }
                }
                found
            }
            Value::Lam { .. } | Value::Pi { .. } => false,
            Value::Thunk { .. } => unreachable!("nb_is_open: thunk after forcing"),
        };
        self.ctx.nb.open_cache.insert(v, r);
        r
    }

    // ---- types of values ----

    /// The type of a value. Defined for everything a conversion question can
    /// stand on: a neutral, a literal or a sort.
    /// The expression a value stands for, folded where the value is: an
    /// unforced constant reads back as the constant applied, and a thunk as
    /// its own expression under its environment. Values reaching inference
    /// hold no conversion-local variables, so a `BVar` head cannot appear.
    pub(crate) fn nb_readback(&mut self, v: ValId) -> ExprPtr<'t> {
        match self.ctx.nb.get(v) {
            Value::Thunk { env, expr, .. } => self.reify(crate::closure::Clo { e: expr, env }),
            Value::NatLit { ptr } => self.ctx.mk_nat_lit(ptr).expect("nat literal"),
            Value::StrLit { ptr } => self.ctx.mk_string_lit(ptr).expect("string literal"),
            Value::Sort { level } => {
                let level = self.ctx.simplify(level);
                self.ctx.mk_sort(level)
            }
            Value::Lam { binder_name, binder_style, binder_type, env, body, .. } => {
                let lam = self.ctx.mk_lambda(binder_name, binder_style, binder_type, body);
                self.reify(crate::closure::Clo { e: lam, env })
            }
            Value::Pi { binder_name, binder_style, domain, env, body } => {
                // the read-back domain has no loose variables, so reifying
                // the rebuilt binder touches only the body
                let d = self.nb_readback(domain);
                let pi = self.ctx.mk_pi(binder_name, binder_style, d, body);
                self.reify(crate::closure::Clo { e: pi, env })
            }
            Value::Unfold { name, levels, spine, .. } => {
                let head = self.ctx.mk_const(name, levels);
                self.nb_readback_spine(head, spine)
            }
            Value::Rigid { head, spine } => {
                let head_e = match head {
                    RigidHead::Local(e) => e,
                    RigidHead::Const(_, name, levels) => self.ctx.mk_const(name, levels),
                    RigidHead::BVar(..) => {
                        unreachable!("conversion-local variable read back")
                    }
                };
                self.nb_readback_spine(head_e, spine)
            }
        }
    }

    fn nb_readback_spine(&mut self, mut out: ExprPtr<'t>, spine: SpineId) -> ExprPtr<'t> {
        let elims = self.ctx.nb.spine_to_vec(spine);
        for elim in elims {
            out = match elim {
                Elim::App(a) => {
                    let a = self.nb_readback(a);
                    self.ctx.mk_app(out, a)
                }
                Elim::Proj { ty_name, idx } => self.ctx.mk_proj(ty_name, idx, out),
            };
        }
        out
    }

    pub(crate) fn nb_type(&mut self, depth: u32, v: ValId) -> ValId {
        let v = self.nb_force(depth, v);
        if let Some(&t) = self.ctx.nb.type_cache.get(&v) {
            return t;
        }
        let t = self.nb_type_go(depth, v);
        self.ctx.nb.type_cache.insert(v, t);
        t
    }

    fn nb_type_go(&mut self, depth: u32, v: ValId) -> ValId {
        match self.ctx.nb.get(v) {
            Value::Sort { level } => {
                let s = self.ctx.succ(level);
                let s = self.ctx.simplify(s);
                self.ctx.nb.mk_sort(s)
            }
            Value::NatLit { .. } => {
                let n = self.ctx.export_file.name_cache.nat.expect("nb_type: Nat");
                let levels = self.ctx.alloc_levels_slice(&[]);
                self.ctx.nb.mk_rigid(RigidHead::Const(ConstKind::Inductive, n, levels), SPINE_EMPTY)
            }
            Value::StrLit { .. } => {
                let n = self.ctx.export_file.name_cache.string.expect("nb_type: String");
                let levels = self.ctx.alloc_levels_slice(&[]);
                self.ctx.nb.mk_rigid(RigidHead::Const(ConstKind::Inductive, n, levels), SPINE_EMPTY)
            }
            Value::Rigid { head, spine } => {
                let head_ty = self.nb_head_type(depth, head);
                let base = self.ctx.nb.mk_rigid(head, SPINE_EMPTY);
                self.nb_spine_type(depth, head_ty, base, spine)
            }
            Value::Unfold { name, levels, spine, .. } => {
                let head_ty = self.nb_const_type(name, levels);
                let base = self.ctx.nb.mk_unfold(name, levels, SPINE_EMPTY);
                self.nb_spine_type(depth, head_ty, base, spine)
            }
            Value::Pi { .. } | Value::Lam { .. } => panic!("nb_type: binder"),
            Value::Thunk { .. } => unreachable!("nb_type: thunk after forcing"),
        }
    }

    fn nb_head_type(&mut self, depth: u32, head: RigidHead<'t>) -> ValId {
        match head {
            RigidHead::BVar(_, ty) => ty,
            RigidHead::Local(e) => {
                let Local { binder_type, .. } = self.ctx.read_expr(e) else {
                    panic!("nb_head_type: not a local")
                };
                self.nb_eval(depth, VENV_NIL, binder_type)
            }
            RigidHead::Const(_, n, ls) => self.nb_const_type(n, ls),
        }
    }

    /// The type a constant's declaration gives it, with its level parameters
    /// replaced, evaluated once per constant.
    pub(crate) fn nb_const_type(&mut self, name: NamePtr<'t>, levels: LevelsPtr<'t>) -> ValId {
        if let Some(&v) = self.ctx.nb.const_ty_cache.get(&(name, levels)) {
            return v;
        }
        let info = *self.env.get_declar(&name).expect("nb_const_type: unknown constant").info();
        let ty = self.ctx.subst_expr_levels(info.ty, info.uparams, levels);
        let v = self.nb_eval(0, VENV_NIL, ty);
        self.ctx.nb.const_ty_cache.insert((name, levels), v);
        v
    }

    /// The type of `base` after the eliminations in `spine`: an application
    /// instantiates the pi's codomain, a projection reads the corresponding
    /// field's type off the structure's constructor.
    fn nb_spine_type(
        &mut self,
        depth: u32,
        mut ty: ValId,
        base: ValId,
        spine: SpineId,
    ) -> ValId {
        let mut prev = base;
        for elim in self.ctx.nb.spine_to_vec(spine) {
            match elim {
                Elim::App(a) => {
                    let ty_f = self.nb_whnf(depth, ty);
                    if !matches!(self.ctx.nb.get(ty_f), Value::Pi { .. }) {
                        panic!("nb_spine_type: applied a non-function");
                    }
                    ty = self.nb_open(depth, ty_f, a);
                    prev = self.nb_apply(depth, prev, a);
                }
                Elim::Proj { ty_name, idx } => {
                    ty = self
                        .nb_field_type(depth, prev, ty, ty_name, idx)
                        .expect("nb_spine_type: projected a non-structure");
                    prev = self.nb_proj(depth, ty_name, idx, prev);
                }
            }
        }
        ty
    }

    /// The type of field `idx` of `struct_value`, whose type is `struct_ty`.
    /// Earlier fields are supplied as projections of `struct_value`, since a
    /// later field's type may mention them.
    fn nb_field_type(
        &mut self,
        depth: u32,
        struct_value: ValId,
        struct_ty: ValId,
        ty_name: NamePtr<'t>,
        idx: usize,
    ) -> Option<ValId> {
        let struct_ty = self.nb_whnf(depth, struct_ty);
        let (ind_name, ind_levels, args) = self.nb_as_inductive(struct_ty)?;
        if ind_name != ty_name {
            return None;
        }
        let ind = self.env.get_structure(&ind_name, true)?;
        let num_params = usize::from(ind.num_params);
        let ctor_name = *ind.all_ctor_names.first()?;
        let ctor_info = *self.env.get_declar(&ctor_name)?.info();
        let ctor_ty = self.ctx.subst_expr_levels(ctor_info.ty, ctor_info.uparams, ind_levels);
        let mut cur = self.nb_eval(depth, VENV_NIL, ctor_ty);
        for i in 0..num_params {
            let cur_f = self.nb_whnf(depth, cur);
            if !matches!(self.ctx.nb.get(cur_f), Value::Pi { .. }) {
                return None;
            }
            cur = self.nb_open(depth, cur_f, *args.get(i)?);
        }
        for i in 0..idx {
            let cur_f = self.nb_whnf(depth, cur);
            if !matches!(self.ctx.nb.get(cur_f), Value::Pi { .. }) {
                return None;
            }
            let prior = self.nb_proj(depth, ty_name, i, struct_value);
            cur = self.nb_open(depth, cur_f, prior);
        }
        let cur_f = self.nb_whnf(depth, cur);
        match self.ctx.nb.get(cur_f) {
            Value::Pi { domain, .. } => Some(self.nb_force(depth, domain)),
            _ => None,
        }
    }

    /// The universe a type lives in, when it can be read off without
    /// inferring the type of the type.
    pub(crate) fn nb_type_level(&mut self, depth: u32, ty: ValId) -> Option<LevelPtr<'t>> {
        let ty = self.nb_force(depth, ty);
        match self.ctx.nb.get(ty) {
            Value::Sort { level } => {
                let s = self.ctx.succ(level);
                Some(self.ctx.simplify(s))
            }
            Value::Pi { domain, .. } => {
                let l_dom = self.nb_type_level(depth, domain)?;
                let dom = self.nb_force(depth, domain);
                let fresh = self.ctx.nb.mk_bvar(depth, dom);
                let cod = self.nb_open(depth + 1, ty, fresh);
                let cod = self.nb_whnf(depth + 1, cod);
                let l_cod = self.nb_type_level(depth + 1, cod)?;
                let l = self.ctx.imax(l_dom, l_cod);
                Some(self.ctx.simplify(l))
            }
            Value::Rigid { head: RigidHead::Const(_, n, ls), .. } => {
                if let Some(l) = self.nb_const_level(n, ls) {
                    return Some(l);
                }
                self.nb_sort_of(depth, ty)
            }
            Value::Unfold { name, levels, .. } => {
                if let Some(l) = self.nb_sort_of(depth, ty) {
                    return Some(l);
                }
                self.nb_const_level(name, levels)
            }
            Value::Rigid { .. } => self.nb_sort_of(depth, ty),
            _ => None,
        }
    }

    fn nb_sort_of(&mut self, depth: u32, ty: ValId) -> Option<LevelPtr<'t>> {
        let t = self.nb_type(depth, ty);
        let t = self.nb_whnf(depth, t);
        match self.ctx.nb.get(t) {
            Value::Sort { level } => Some(self.ctx.simplify(level)),
            _ => None,
        }
    }

    /// The universe a constant's type ends in, walking off its telescope.
    fn nb_const_level(
        &mut self,
        name: NamePtr<'t>,
        levels: LevelsPtr<'t>,
    ) -> Option<LevelPtr<'t>> {
        if let Some(&l) = self.ctx.nb.const_lvl_cache.get(&(name, levels)) {
            return l;
        }
        let mut cur = self.nb_const_type(name, levels);
        let mut d = 0u32;
        let r = loop {
            let cur_f = self.nb_whnf(d, cur);
            match self.ctx.nb.get(cur_f) {
                Value::Pi { domain, .. } => {
                    let dom = self.nb_force(d, domain);
                    let fresh = self.ctx.nb.mk_bvar(d, dom);
                    cur = self.nb_open(d + 1, cur_f, fresh);
                    d += 1;
                }
                Value::Sort { level } => break Some(self.ctx.simplify(level)),
                _ => break None,
            }
        };
        self.ctx.nb.const_lvl_cache.insert((name, levels), r);
        r
    }
}
