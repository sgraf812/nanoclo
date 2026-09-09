//! Conversion on values.
//!
//! Two values are compared by walking them together. Because every value is
//! interned, the first test is that their indices are equal, which settles
//! every pair that shares any construction history: the same definition
//! unfolded twice, the same argument evaluated under two spellings of one
//! environment, the same recursor step reached along two routes. Only when
//! that fails does the comparison reduce.
//!
//! `RIGID` says whether the comparison is allowed to reduce. A comparison of
//! the arguments of two applications of one constant is a guess: the
//! arguments may differ while the applications are still equal, so that
//! comparison runs with `RIGID` off, settling only what identity and
//! congruence settle, and its failures are recorded separately and dropped
//! with the guess.

use crate::tc::SPEC_BUDGET;
use crate::env::{Declar, ReducibilityHint};
use crate::nbe::{ConstKind, Elim, RigidHead, SpineId, ValId, Value};
use crate::tc::TypeChecker;
use crate::util::NamePtr;

/// Whether a comparison is worth recording. A pair of neutrals with the same
/// head is settled by their arguments, which are recorded on their own; the
/// pairs worth a table entry are the ones whose answer took reduction.
fn is_cacheable(v: &Value<'_>) -> bool {
    matches!(
        v,
        Value::Pi { .. }
            | Value::Lam { .. }
            | Value::Unfold { .. }
            | Value::Rigid {
                head: RigidHead::Const(ConstKind::Recursor | ConstKind::QuotConst, ..),
                ..
            }
    )
}

/// Whether a neutral standing on this head can still fire.
fn is_iota_head(h: RigidHead<'_>) -> bool {
    matches!(h, RigidHead::Const(ConstKind::Recursor | ConstKind::QuotConst, ..))
}

fn head_eq(a: RigidHead<'_>, b: RigidHead<'_>) -> bool {
    match (a, b) {
        (RigidHead::BVar(x, _), RigidHead::BVar(y, _)) => x == y,
        (RigidHead::Local(x), RigidHead::Local(y)) => x == y,
        _ => false,
    }
}

impl<'x, 't: 'x, 'p: 't> TypeChecker<'x, 't, 'p> {
    pub(crate) fn nb_conv(&mut self, depth: u32, x: ValId, y: ValId) -> bool {
        self.ctx.nb.in_conv += 1;
        let r = self.nb_unify::<true>(depth, x, y);
        self.ctx.nb.in_conv -= 1;
        r
    }

    fn nb_unify<const RIGID: bool>(&mut self, depth: u32, x: ValId, y: ValId) -> bool {
        self.ctx.rp.ctrs[11] += 1;
        let x = self.nb_force(depth, x);
        let y = self.nb_force(depth, y);
        if x == y {
            return true;
        }
        if self.ctx.nb.probe_depth > 0 {
            if self.ctx.nb.probe_aborted {
                return false;
            }
            if self.ctx.nb.probe_fuel == 0 {
                self.ctx.nb.probe_aborted = true;
                self.ctx.rp.ctrs[23] += 1;
                return false;
            }
            self.ctx.nb.probe_fuel -= 1;
        }
        self.nb_unify_cached::<RIGID>(depth, x, y)
    }

    fn nb_unify_cached<const RIGID: bool>(&mut self, depth: u32, x: ValId, y: ValId) -> bool {
        let cacheable =
            is_cacheable(&self.ctx.nb.get(x)) || is_cacheable(&self.ctx.nb.get(y));
        if !cacheable {
            return self.nb_unify_go::<RIGID>(depth, x, y);
        }
        let key = if x < y { (x, y) } else { (y, x) };
        if self.ctx.nb.conv_pos.contains(&key) {
            return true;
        }
        // A lambda pair can still be equal after eta expansion, so a failure
        // to match them directly is not a failure of the pair.
        let neg_eligible = !matches!(self.ctx.nb.get(x), Value::Lam { .. })
            && !matches!(self.ctx.nb.get(y), Value::Lam { .. });
        if RIGID && neg_eligible {
            if self.ctx.nb.conv_neg.contains(&key) {
                return false;
            }
            if self.ctx.nb.probe_depth > 0 && self.ctx.nb.conv_neg_probe.contains(&key) {
                return false;
            }
        }
        let r = self.nb_unify_go::<RIGID>(depth, x, y);
        if self.ctx.nb.probe_aborted {
            return r;
        }
        if r {
            self.ctx.nb.conv_pos.insert(key);
        } else if RIGID && neg_eligible {
            if self.ctx.nb.probe_depth == 0 {
                self.ctx.nb.conv_neg.insert(key);
            } else {
                self.ctx.nb.conv_neg_probe.insert(key);
            }
        }
        r
    }

    fn nb_unify_go<const RIGID: bool>(&mut self, depth: u32, x: ValId, y: ValId) -> bool {
        if let Some(r) = self.nb_conv_nat::<RIGID>(depth, x, y) {
            return r;
        }
        if let Some(r) = self.nb_conv_str::<RIGID>(depth, x, y) {
            return r;
        }
        if self.nb_unify_direct::<RIGID>(depth, x, y) {
            return true;
        }
        if !RIGID {
            return false;
        }
        if self.nb_proof_irrel(depth, x, y) {
            return true;
        }
        // eta: a lambda against something that is not one
        match (self.ctx.nb.get(x), self.ctx.nb.get(y)) {
            (Value::Lam { .. }, ref o) if !matches!(o, Value::Lam { .. }) => {
                let domain = self.nb_lam_domain(depth, x);
                let fresh = self.ctx.nb.mk_bvar(depth, domain);
                let lhs = self.nb_open(depth + 1, x, fresh);
                let rhs = self.nb_apply(depth + 1, y, fresh);
                return self.nb_unify::<true>(depth + 1, lhs, rhs);
            }
            (ref o, Value::Lam { .. }) if !matches!(o, Value::Lam { .. }) => {
                let domain = self.nb_lam_domain(depth, y);
                let fresh = self.ctx.nb.mk_bvar(depth, domain);
                let lhs = self.nb_apply(depth + 1, x, fresh);
                let rhs = self.nb_open(depth + 1, y, fresh);
                return self.nb_unify::<true>(depth + 1, lhs, rhs);
            }
            _ => {}
        }
        self.nb_struct_eta(depth, x, y)
    }

    fn nb_unify_direct<const RIGID: bool>(&mut self, depth: u32, x: ValId, y: ValId) -> bool {
        let (vx, vy) = (self.ctx.nb.get(x), self.ctx.nb.get(y));
        match (vx, vy) {
            (Value::Sort { level: lx }, Value::Sort { level: ly }) => {
                self.ctx.eq_antisymm(lx, ly)
            }
            (Value::NatLit { ptr: px }, Value::NatLit { ptr: py }) => px == py,
            (Value::StrLit { ptr: px }, Value::StrLit { ptr: py }) => px == py,

            (Value::Pi { domain: dx, .. }, Value::Pi { domain: dy, .. }) => {
                if !self.nb_unify::<RIGID>(depth, dx, dy) {
                    return false;
                }
                let dom = self.nb_force(depth, dx);
                let fresh = self.ctx.nb.mk_bvar(depth, dom);
                let bx = self.nb_open(depth + 1, x, fresh);
                let by = self.nb_open(depth + 1, y, fresh);
                self.nb_unify::<RIGID>(depth + 1, bx, by)
            }

            (Value::Lam { .. }, Value::Lam { .. }) => {
                let dom = self.nb_lam_domain(depth, x);
                let fresh = self.ctx.nb.mk_bvar(depth, dom);
                let bx = self.nb_open(depth + 1, x, fresh);
                let by = self.nb_open(depth + 1, y, fresh);
                self.nb_unify::<RIGID>(depth + 1, bx, by)
            }

            (
                Value::Rigid { head: hx, spine: sx },
                Value::Rigid { head: hy, spine: sy },
            ) => {
                let const_heads_match = match (hx, hy) {
                    (RigidHead::Const(kx, nx, lx), RigidHead::Const(ky, ny, ly)) => {
                        kx == ky && nx == ny && self.ctx.eq_antisymm_many(lx, ly)
                    }
                    _ => false,
                };
                // Either side standing on a recursor may still reduce, so the
                // spines decide only after neither side steps.
                if is_iota_head(hx) || is_iota_head(hy) {
                    return self.nb_unify_iota::<RIGID>(depth, x, y, const_heads_match, sx, sy);
                }
                if matches!(hx, RigidHead::Const(..)) || matches!(hy, RigidHead::Const(..)) {
                    return const_heads_match && self.nb_unify_spine::<RIGID>(depth, sx, sy);
                }
                head_eq(hx, hy) && self.nb_unify_spine::<RIGID>(depth, sx, sy)
            }

            (
                Value::Unfold { name: nx, levels: lx, spine: sx, .. },
                Value::Unfold { name: ny, levels: ly, spine: sy, .. },
            ) => {
                let heads_match = nx == ny && self.ctx.eq_antisymm_many(lx, ly);
                if !RIGID {
                    return heads_match && self.nb_unify_spine::<false>(depth, sx, sy);
                }
                if heads_match && self.nb_spine_probe(depth, sx, sy) {
                    return true;
                }
                if self.nb_proof_irrel(depth, x, y) {
                    return true;
                }
                if heads_match {
                    let r = self.nb_unfold_pair(depth, x, y);
                    self.ctx.nb.probe_escalate = 0;
                    return r;
                }
                // Unfold the one whose definition is nearer the leaves, so
                // the two meet at a shared subterm rather than at normal
                // forms.
                let hx = self.nb_hint(nx);
                let hy = self.nb_hint(ny);
                if hx.is_lt(&hy) {
                    self.nb_unfold_one::<false>(depth, x, y)
                } else if hy.is_lt(&hx) {
                    self.nb_unfold_one::<true>(depth, x, y)
                } else {
                    self.nb_unfold_pair(depth, x, y)
                }
            }

            (Value::Unfold { .. }, _) if RIGID => {
                if self.nb_proof_irrel(depth, x, y) {
                    return true;
                }
                let mut x2 = self.nb_unfold(depth, x);
                if x2 == x {
                    x2 = self.nb_unfold_demand(depth, x);
                    if x2 == x {
                        return false;
                    }
                }
                self.nb_unify::<true>(depth, x2, y)
            }
            (_, Value::Unfold { .. }) if RIGID => {
                if self.nb_proof_irrel(depth, x, y) {
                    return true;
                }
                let mut y2 = self.nb_unfold(depth, y);
                if y2 == y {
                    y2 = self.nb_unfold_demand(depth, y);
                    if y2 == y {
                        return false;
                    }
                }
                self.nb_unify::<true>(depth, x, y2)
            }

            (Value::Rigid { head: RigidHead::Const(k, ..), spine: sx }, _)
                if RIGID && matches!(k, ConstKind::Recursor | ConstKind::QuotConst) =>
            {
                let _ = sx;
                if self.nb_proof_irrel(depth, x, y) {
                    return true;
                }
                match self.nb_iota(depth, x) {
                    Some(x2) => self.nb_unify::<true>(depth, x2, y),
                    None => false,
                }
            }
            (_, Value::Rigid { head: RigidHead::Const(k, ..), .. })
                if RIGID && matches!(k, ConstKind::Recursor | ConstKind::QuotConst) =>
            {
                if self.nb_proof_irrel(depth, x, y) {
                    return true;
                }
                match self.nb_iota(depth, y) {
                    Some(y2) => self.nb_unify::<true>(depth, x, y2),
                    None => false,
                }
            }

            _ => false,
        }
    }

    /// Unfold `x` if `LEFT`, else `y`, and compare again. If that side turns
    /// out not to unfold, try the other.
    fn nb_unfold_one<const LEFT: bool>(&mut self, depth: u32, x: ValId, y: ValId) -> bool {
        let (first, second) = if LEFT { (x, y) } else { (y, x) };
        let f2 = self.nb_unfold(depth, first);
        if f2 != first {
            return if LEFT {
                self.nb_unify::<true>(depth, f2, y)
            } else {
                self.nb_unify::<true>(depth, x, f2)
            };
        }
        let s2 = self.nb_unfold(depth, second);
        if s2 != second {
            return if LEFT {
                self.nb_unify::<true>(depth, first, s2)
            } else {
                self.nb_unify::<true>(depth, s2, first)
            };
        }
        let d2 = self.nb_unfold_demand(depth, first);
        if d2 == first {
            return false;
        }
        if LEFT {
            self.nb_unify::<true>(depth, d2, y)
        } else {
            self.nb_unify::<true>(depth, x, d2)
        }
    }

    fn nb_unfold_pair(&mut self, depth: u32, x: ValId, y: ValId) -> bool {
        let x2 = self.nb_unfold(depth, x);
        let y2 = self.nb_unfold(depth, y);
        if x2 == x && y2 == y {
            let f1 = self.nb_unfold_demand(depth, x);
            let f2 = self.nb_unfold_demand(depth, y);
            if f1 == x && f2 == y {
                return false;
            }
            return self.nb_unify::<true>(depth, f1, f2);
        }
        self.nb_unify::<true>(depth, x2, y2)
    }

    fn nb_unify_iota<const RIGID: bool>(
        &mut self,
        depth: u32,
        x: ValId,
        y: ValId,
        heads_match: bool,
        sx: SpineId,
        sy: SpineId,
    ) -> bool {
        if !RIGID {
            return heads_match && self.nb_unify_spine::<false>(depth, sx, sy);
        }
        if heads_match && self.nb_spine_probe(depth, sx, sy) {
            return true;
        }
        if self.nb_proof_irrel(depth, x, y) {
            return true;
        }
        let x2 = self.nb_iota(depth, x).unwrap_or(x);
        let y2 = self.nb_iota(depth, y).unwrap_or(y);
        if x2 != x || y2 != y {
            let r = self.nb_unify::<true>(depth, x2, y2);
            self.ctx.nb.probe_escalate = 0;
            return r;
        }
        heads_match && self.nb_unify_spine::<true>(depth, sx, sy)
    }

    /// Compare the arguments of two applications of one constant, recording
    /// failures apart so that they are dropped when the guess is.
    fn nb_spine_probe(&mut self, depth: u32, sx: SpineId, sy: SpineId) -> bool {
        let key = if sx < sy { (sx, sy) } else { (sy, sx) };
        if self.ctx.nb.probe_fail.contains(&key) {
            self.ctx.rp.ctrs[25] += 1;
            return false;
        }
        let outer = self.ctx.nb.probe_depth == 0;
        let mut granted = 0;
        if outer {
            // An aborted probe hands its doubled grant one step down the
            // unfold chain; an independent probe starts at the base again.
            granted = if self.ctx.nb.probe_escalate > 0 {
                std::mem::take(&mut self.ctx.nb.probe_escalate)
            } else {
                SPEC_BUDGET
            };
            self.ctx.nb.probe_fuel = granted;
            self.ctx.nb.probe_aborted = false;
        }
        self.ctx.nb.probe_depth += 1;
        let r = self.nb_unify_spine::<true>(depth, sx, sy);
        self.ctx.nb.probe_depth -= 1;
        if self.ctx.nb.probe_depth == 0 {
            if !self.ctx.nb.conv_neg_probe.is_empty() {
                self.ctx.nb.conv_neg_probe.clear();
            }
            // an exhausted comparison produced no answer
            if self.ctx.nb.probe_aborted {
                self.ctx.nb.probe_aborted = false;
                self.ctx.rp.ctrs[24] += 1;
                self.ctx.nb.probe_fail.insert(key);
                const ESCALATE_CAP: u64 = 1 << 20;
                self.ctx.nb.probe_escalate = (granted * 2).min(ESCALATE_CAP);
                return false;
            }
            if !r {
                self.ctx.nb.probe_fail.insert(key);
            }
        }
        r
    }

    fn nb_unify_spine<const RIGID: bool>(&mut self, depth: u32, sx: SpineId, sy: SpineId) -> bool {
        if sx == sy {
            return true;
        }
        if self.ctx.nb.spine_len(sx) != self.ctx.nb.spine_len(sy) {
            return false;
        }
        self.nb_unify_spine_go::<RIGID>(depth, sx, sy)
    }

    /// Walk two equal-length spines innermost elimination first. Spines are
    /// interned, so a shared prefix is the same spine and one comparison
    /// settles all of it.
    fn nb_unify_spine_go<const RIGID: bool>(&mut self, depth: u32, sx: SpineId, sy: SpineId) -> bool {
        if sx == sy {
            return true;
        }
        let nx = &self.ctx.nb.spines[sx as usize];
        let ny = &self.ctx.nb.spines[sy as usize];
        let (ex, ey) = (nx.elim, ny.elim);
        let (px, py) = (nx.parent, ny.parent);
        if !self.nb_unify_spine_go::<RIGID>(depth, px, py) {
            return false;
        }
        match (ex, ey) {
            (Elim::App(a), Elim::App(b)) => self.nb_unify::<RIGID>(depth, a, b),
            (
                Elim::Proj { ty_name: tx, idx: ix },
                Elim::Proj { ty_name: ty, idx: iy },
            ) => tx == ty && ix == iy,
            _ => false,
        }
    }

    fn nb_hint(&self, name: NamePtr<'t>) -> ReducibilityHint {
        match self.env.get_declar(&name) {
            Some(Declar::Definition { hint, .. }) => *hint,
            _ => ReducibilityHint::Opaque,
        }
    }

    // ---- proof irrelevance ----

    /// Two proofs of one proposition are equal.
    fn nb_proof_irrel(&mut self, depth: u32, x: ValId, y: ValId) -> bool {
        let (vx, vy) = (self.ctx.nb.get(x), self.ctx.nb.get(y));
        if matches!(vx, Value::Lam { .. }) || matches!(vy, Value::Lam { .. }) {
            return self.nb_proof_irrel_lam(depth, x, y);
        }
        if !matches!(vx, Value::Rigid { .. } | Value::Unfold { .. }) {
            return false;
        }
        if !matches!(vy, Value::Rigid { .. } | Value::Unfold { .. }) {
            return false;
        }
        let tx = self.nb_type(depth, x);
        if !self.nb_is_prop(depth, tx) {
            return false;
        }
        let ty = self.nb_type(depth, y);
        if !self.nb_is_prop(depth, ty) {
            return false;
        }
        self.nb_unify::<true>(depth, tx, ty)
    }

    /// A function into a proposition is pointwise a proof, so compare the
    /// two under a fresh variable.
    fn nb_proof_irrel_lam(&mut self, depth: u32, x: ValId, y: ValId) -> bool {
        let lam = if matches!(self.ctx.nb.get(x), Value::Lam { .. }) { x } else { y };
        let domain = self.nb_lam_domain(depth, lam);
        let fresh = self.ctx.nb.mk_bvar(depth, domain);
        let xb = self.nb_apply(depth + 1, x, fresh);
        let yb = self.nb_apply(depth + 1, y, fresh);
        self.nb_proof_irrel(depth + 1, xb, yb)
    }

    fn nb_is_prop(&mut self, depth: u32, ty: ValId) -> bool {
        match self.nb_type_level(depth, ty) {
            Some(l) => self.ctx.is_zero(l),
            None => false,
        }
    }

    /// Whether the universe of `ty` may be zero: a universe parameter counts,
    /// since an instantiation may send it there.
    pub(crate) fn nb_may_be_prop(&mut self, depth: u32, ty: ValId) -> bool {
        match self.nb_type_level(depth, ty) {
            Some(l) => self.ctx.may_be_prop(l),
            None => false,
        }
    }

    // ---- structures ----

    /// A value of a structure type is the constructor applied to its
    /// projections, and a structure with one field-free constructor has one
    /// value.
    fn nb_struct_eta(&mut self, depth: u32, x: ValId, y: ValId) -> bool {
        for v in [x, y] {
            if matches!(self.ctx.nb.get(v), Value::Pi { .. } | Value::Lam { .. }) {
                continue;
            }
            let ty = self.nb_type(depth, v);
            let ty = self.nb_whnf(depth, ty);
            let Some((ind_name, _, _)) = self.nb_as_inductive(ty) else { continue };
            if self.nb_is_unit(ind_name) {
                return true;
            }
            if !self.env.can_be_struct(&ind_name) {
                continue;
            }
            if self.nb_eta_struct(depth, ind_name, x, y)
                || self.nb_eta_struct(depth, ind_name, y, x)
            {
                return true;
            }
        }
        false
    }

    /// `y` is a constructor application; compare each of its fields with the
    /// corresponding projection of `x`.
    fn nb_eta_struct(&mut self, depth: u32, ind_name: NamePtr<'t>, x: ValId, y: ValId) -> bool {
        let Value::Rigid { head: RigidHead::Const(ConstKind::Ctor, ctor, _), spine } =
            self.ctx.nb.get(y)
        else {
            return false;
        };
        let Some(cd) = self.env.get_constructor(&ctor) else { return false };
        let (inductive_name, num_params, num_fields) =
            (cd.inductive_name, usize::from(cd.num_params), usize::from(cd.num_fields));
        if inductive_name != ind_name {
            return false;
        }
        let Some(args) = self.ctx.nb.spine_args(spine) else { return false };
        if args.len() != num_params + num_fields {
            return false;
        }
        for i in 0..num_fields {
            let proj = self.nb_proj(depth, ind_name, i, x);
            if !self.nb_unify::<true>(depth, proj, args[num_params + i]) {
                return false;
            }
        }
        true
    }

    fn nb_is_unit(&self, ind_name: NamePtr<'t>) -> bool {
        let Some(ind) = self.env.get_structure(&ind_name, false) else { return false };
        let Some(&ctor) = ind.all_ctor_names.first() else { return false };
        match self.env.get_constructor(&ctor) {
            Some(cd) => cd.num_fields == 0,
            None => false,
        }
    }

    // ---- the Nat extension ----

    /// `Nat.succ a` and `Nat.succ b` are equal exactly when `a` and `b` are,
    /// which settles a pair of literals-as-constructors without unfolding
    /// either into a tower.
    fn nb_conv_nat<const RIGID: bool>(
        &mut self,
        depth: u32,
        x: ValId,
        y: ValId,
    ) -> Option<bool> {
        if !self.nb_may_be_nat(x) && !self.nb_may_be_nat(y) {
            return None;
        }
        if matches!(self.ctx.nb.get(x), Value::NatLit { .. })
            && matches!(self.ctx.nb.get(y), Value::NatLit { .. })
        {
            return None;
        }
        let xz = self.nb_is_nat_zero(x);
        let yz = self.nb_is_nat_zero(y);
        if xz && yz {
            return Some(true);
        }
        let px = self.nb_nat_pred(x);
        let py = self.nb_nat_pred(y);
        match (px, py) {
            (Some(a), Some(b)) => Some(self.nb_unify::<RIGID>(depth, a, b)),
            _ => None,
        }
    }

    /// A string literal and the constructor application it denotes stand
    /// for the same string, so `"ab"` meets `String.mk ['a', 'b']` as the
    /// list of characters it spells out.
    fn nb_conv_str<const RIGID: bool>(
        &mut self,
        depth: u32,
        x: ValId,
        y: ValId,
    ) -> Option<bool> {
        let lit_x = matches!(self.ctx.nb.get(x), Value::StrLit { .. });
        let lit_y = matches!(self.ctx.nb.get(y), Value::StrLit { .. });
        if lit_x == lit_y {
            return None;
        }
        let (lit, other) = if lit_x { (x, y) } else { (y, x) };
        if !self.nb_is_string_ctor(other) {
            return None;
        }
        let Value::StrLit { ptr } = self.ctx.nb.get(lit) else {
            return None;
        };
        let c = self.nb_str_to_ctor(depth, ptr)?;
        Some(self.nb_unify::<RIGID>(depth, c, other))
    }

    fn nb_is_string_ctor(&self, v: ValId) -> bool {
        match self.ctx.nb.get(v) {
            Value::Rigid { head: RigidHead::Const(_, name, _), .. }
            | Value::Unfold { name, .. } => {
                let nc = &self.ctx.export_file.name_cache;
                Some(name) == nc.string_mk || Some(name) == nc.string_of_list
            }
            _ => false,
        }
    }

    fn nb_may_be_nat(&self, v: ValId) -> bool {
        match self.ctx.nb.get(v) {
            Value::NatLit { .. } => true,
            Value::Rigid { head: RigidHead::Const(ConstKind::Ctor, name, _), .. } => {
                let nc = &self.ctx.export_file.name_cache;
                Some(name) == nc.nat_zero || Some(name) == nc.nat_succ
            }
            _ => false,
        }
    }

    fn nb_is_nat_zero(&self, v: ValId) -> bool {
        match self.ctx.nb.get(v) {
            Value::Rigid { head: RigidHead::Const(ConstKind::Ctor, name, _), spine } => {
                Some(name) == self.ctx.export_file.name_cache.nat_zero
                    && self.ctx.nb.spine_len(spine) == 0
            }
            Value::NatLit { ptr } => {
                use num_traits::Zero;
                self.ctx.read_bignum(ptr).map(|n| n.is_zero()).unwrap_or(false)
            }
            _ => false,
        }
    }

    fn nb_nat_pred(&mut self, v: ValId) -> Option<ValId> {
        match self.ctx.nb.get(v) {
            Value::Rigid { head: RigidHead::Const(ConstKind::Ctor, name, _), spine } => {
                if Some(name) != self.ctx.export_file.name_cache.nat_succ
                    || self.ctx.nb.spine_len(spine) != 1
                {
                    return None;
                }
                match self.ctx.nb.spine_get(spine, 0) {
                    Some(Elim::App(a)) => Some(a),
                    _ => None,
                }
            }
            Value::NatLit { ptr } => {
                use num_traits::Zero;
                let n = self.ctx.read_bignum(ptr)?.clone();
                if n.is_zero() {
                    return None;
                }
                let p = self.ctx.alloc_bignum(n - 1u8)?;
                Some(self.ctx.nb.mk_nat(p))
            }
            _ => None,
        }
    }
}
