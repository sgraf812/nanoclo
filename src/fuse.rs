//! Wrapper fusion.
//!
//! Lean compiles pattern matching and structural recursion into layers of
//! definitions: `f` unfolds to `f.match_1`, which unfolds to `T.casesOn`, which
//! unfolds to `T.rec`; `T.brecOn` builds a table of `PProd` pairs that the body
//! reads with projections; `a + b` passes through `HAdd.hAdd`, `instHAdd` and
//! `Add.add`. An evaluator that unfolds these one at a time pays several
//! unfoldings and applications for every step of the recursion.
//!
//! `fuse_body` rewrites the level-instantiated value of a definition once,
//! under every binder: it unfolds wrapper heads (matchers, `casesOn`, `brecOn`
//! and small abbreviations), contracts the beta redexes that exposes, fires a
//! recursor whose major premise is syntactically a constructor application,
//! and projects a field out of a constructor application. For example the
//! body of `List.get!Internal` becomes a `List.rec` whose `cons` case holds a
//! `Nat.rec` directly, where the unfused body reaches the same two recursors
//! through `List.brecOn`, a matcher, `_sparseCasesOn_1` and `Nat.casesOn`.
//! Every rewrite is a beta, delta, iota or projection step on a
//! closed term, so the result is definitionally equal to the value. Only the
//! evaluator reads fused bodies; inference never sees them.

use crate::env::{Declar, ReducibilityHint};
use crate::expr::Expr::*;
use crate::name::Name;
use crate::tc::TypeChecker;
use crate::util::{new_fx_hash_map, ExprPtr, FxHashMap, LevelsPtr, NamePtr};

/// Largest body, in distinct nodes, of a wrapper that is unfolded.
const WRAPPER_CAP: usize = 512;
/// Largest body, in distinct nodes, of an abbreviation that is unfolded.
const ABBREV_CAP: usize = 256;
/// Nodes one fusion may visit.
const BUDGET: usize = 1 << 18;
/// Nested reductions one fusion may perform.
const MAX_DEPTH: u32 = 48;
/// Tree nodes one fusion may duplicate by substitution.
const DUP_BUDGET: usize = 1 << 15;
/// Constants unfolded at one head position.
const MAX_UNFOLDS: u32 = 64;
/// Occurrences counted before every variable is taken as duplicated.
const COUNT_VISITS: usize = 20_000;

struct Fuser<'t> {
    memo: FxHashMap<ExprPtr<'t>, ExprPtr<'t>>,
    budget: usize,
    depth: u32,
    dup_budget: usize,
    overflow: bool,
}

impl<'x, 't: 'x, 'p: 't> TypeChecker<'x, 't, 'p> {
    /// The fused form of `e`, the closed value of a definition.
    fn fuse_body(&mut self, e: ExprPtr<'t>) -> ExprPtr<'t> {
        let saved = self.ctx.dbj_level_counter;
        let mut st = Fuser {
            memo: new_fx_hash_map(),
            budget: BUDGET,
            depth: 0,
            dup_budget: DUP_BUDGET,
            overflow: false,
        };
        let r = self.fz(&mut st, e);
        self.ctx.dbj_level_counter = saved;
        r
    }

    /// The fused value of the definition `name` at `levels`, whose
    /// level-instantiated value is `val`. Shared across declarations under
    /// the lifetime discipline of `g_unfold`.
    pub(crate) fn fused_value(
        &mut self,
        name: NamePtr<'t>,
        levels: LevelsPtr<'t>,
        val: ExprPtr<'t>,
    ) -> ExprPtr<'t> {
        if self.env.has_temp_ext() {
            return self.fuse_body(val);
        }
        let key = self.ctx.mk_const(name, levels);
        if let Some(&r) = self.ctx.rp.g_fused.get(&key) {
            return r;
        }
        let r = self.fuse_body(val);
        self.ctx.rp.g_fused.insert(key, r);
        r
    }

    /// Whether `name` is a definition the equation compiler generates to
    /// implement matching or structural recursion.
    fn is_recursion_wrapper(&mut self, name: NamePtr<'t>) -> bool {
        if let Some(&w) = self.ctx.rp.fuse_wrapper.get(&name) {
            return w;
        }
        let w = match self.ctx.read_name(name) {
            Name::Str(pfx, sfx, _) => {
                let s: &str = self.ctx.read_string(sfx);
                matches!(s, "casesOn" | "recOn" | "brecOn" | "binductionOn" | "_f")
                    || s.starts_with("match_")
                    || s.starts_with("_sparseCasesOn")
                    || (s == "go"
                        && matches!(self.ctx.read_name(pfx),
                            Name::Str(_, p, _) if &**self.ctx.read_string(p) == "brecOn"))
            }
            _ => false,
        };
        self.ctx.rp.fuse_wrapper.insert(name, w);
        w
    }

    /// Whether fusion unfolds `name`: a wrapper or an abbreviation with a
    /// small body, other than the primitives the evaluator computes itself.
    fn fuse_unfoldable(&mut self, name: NamePtr<'t>) -> bool {
        if let Some(&u) = self.ctx.rp.fuse_unfoldable.get(&name) {
            return u;
        }
        let env = self.env;
        let u = match env.get_declar(&name) {
            Some(Declar::Definition { val, hint, .. })
                if *hint != ReducibilityHint::Opaque
                    && !self.nb_is_nat_prim(name)
                    && Some(name) != self.ctx.export_file.name_cache.string_of_list =>
            {
                let cap = if self.is_recursion_wrapper(name) {
                    WRAPPER_CAP
                } else if *hint == ReducibilityHint::Abbrev {
                    ABBREV_CAP
                } else {
                    0
                };
                cap > 0 && self.body_size(*val, cap) <= cap
            }
            _ => false,
        };
        self.ctx.rp.fuse_unfoldable.insert(name, u);
        u
    }

    /// Distinct nodes of `e` under its lambda prefix, counted up to past `cap`.
    fn body_size(&self, mut e: ExprPtr<'t>, cap: usize) -> usize {
        while let Lambda { body, .. } = self.ctx.read_expr(e) {
            e = body;
        }
        let mut seen = crate::util::new_fx_hash_set();
        let mut todo = vec![e];
        while let Some(x) = todo.pop() {
            if seen.len() > cap {
                break;
            }
            if !seen.insert(x) {
                continue;
            }
            push_children(self.ctx.read_expr(x), &mut todo);
        }
        seen.len()
    }

    /// Tree nodes of `e`, counted up to past `cap`.
    fn tree_size(&self, e: ExprPtr<'t>, cap: usize) -> usize {
        let mut n = 0;
        let mut todo = vec![e];
        while let Some(x) = todo.pop() {
            if n > cap {
                break;
            }
            n += 1;
            push_children(self.ctx.read_expr(x), &mut todo);
        }
        n
    }

    /// An argument that may be duplicated without losing evaluation sharing:
    /// an atom, a lambda, or a constructor application of such.
    fn is_cheap_arg(&self, a: ExprPtr<'t>, depth: u32) -> bool {
        match self.ctx.read_expr(a) {
            Var { .. } | Local { .. } | Const { .. } | Sort { .. } | NatLit { .. } | StringLit { .. } => true,
            Lambda { .. } => true,
            App { .. } => {
                if depth > 3 {
                    return false;
                }
                let (h, args) = self.ctx.unfold_apps(a);
                let Const { name, .. } = self.ctx.read_expr(h) else { return false };
                let Some(c) = self.env.get_constructor(&name) else { return false };
                args.len() == usize::from(c.num_params + c.num_fields)
                    && args.iter().all(|&x| self.is_cheap_arg(x, depth + 1))
            }
            _ => false,
        }
    }

    /// Occurrences, capped at 2, of the loose variables `0..n` of `b`.
    fn count_bvars(&self, b: ExprPtr<'t>, n: usize) -> Vec<u8> {
        let mut cnt = vec![0u8; n];
        let mut todo = vec![(b, 0u16)];
        let mut visits = 0;
        while let Some((x, d)) = todo.pop() {
            if self.ctx.num_loose_bvars(x) <= d {
                continue;
            }
            visits += 1;
            if visits > COUNT_VISITS {
                cnt.iter_mut().for_each(|c| *c = 2);
                return cnt;
            }
            match self.ctx.read_expr(x) {
                Var { dbj_idx, .. } => {
                    let j = usize::from(dbj_idx - d);
                    if j < n && cnt[j] < 2 {
                        cnt[j] += 1;
                    }
                }
                App { fun, arg, .. } => {
                    todo.push((fun, d));
                    todo.push((arg, d));
                }
                Lambda { binder_type, body, .. } | Pi { binder_type, body, .. } => {
                    todo.push((binder_type, d));
                    todo.push((body, d + 1));
                }
                Let { binder_type, val, body, .. } => {
                    todo.push((binder_type, d));
                    todo.push((val, d));
                    todo.push((body, d + 1));
                }
                Proj { structure, .. } => todo.push((structure, d)),
                _ => {}
            }
        }
        cnt
    }

    /// Whether contracting `(fun x_1 .. x_n => b) args` duplicates no
    /// argument that costs evaluation, within the duplication budget.
    fn share_ok(&self, st: &mut Fuser<'t>, b: ExprPtr<'t>, args: &[ExprPtr<'t>]) -> bool {
        let n = args.len();
        let costly = |a: ExprPtr<'t>| {
            !self.is_cheap_arg(a, 0) || matches!(self.ctx.read_expr(a), Lambda { .. } | App { .. })
        };
        if !args.iter().any(|&a| costly(a)) {
            return true;
        }
        let cnt = self.count_bvars(b, n);
        let mut cost = 0;
        for (j, &a) in args.iter().enumerate() {
            if cnt[n - 1 - j] < 2 {
                continue;
            }
            if !self.is_cheap_arg(a, 0) {
                return false;
            }
            if matches!(self.ctx.read_expr(a), Lambda { .. } | App { .. }) {
                cost += self.tree_size(a, st.dup_budget + 1);
            }
        }
        if cost > st.dup_budget {
            return false;
        }
        st.dup_budget -= cost;
        true
    }

    /// Field `idx` of `s` when `s` is a full application of a constructor of
    /// `ty_name`.
    fn ctor_app_field(&self, s: ExprPtr<'t>, ty_name: NamePtr<'t>, idx: usize) -> Option<ExprPtr<'t>> {
        let (h, args) = self.ctx.unfold_apps(s);
        let Const { name, .. } = self.ctx.read_expr(h) else { return None };
        let c = self.env.get_constructor(&name)?;
        let np = usize::from(c.num_params);
        (c.inductive_name == ty_name
            && args.len() == np + usize::from(c.num_fields)
            && idx < usize::from(c.num_fields))
        .then(|| args[np + idx])
    }

    /// The iota reduct of `rec args` when its major premise is a full
    /// constructor application.
    fn try_iota(
        &mut self,
        rec_name: NamePtr<'t>,
        levels: LevelsPtr<'t>,
        args: &[ExprPtr<'t>],
    ) -> Option<ExprPtr<'t>> {
        let env = self.env;
        let rec = env.get_recursor(&rec_name)?;
        let major_idx = rec.major_idx();
        let major = *args.get(major_idx)?;
        let (mh, margs) = self.ctx.unfold_apps(major);
        let Const { name: ctor_name, .. } = self.ctx.read_expr(mh) else { return None };
        let ctor = env.get_constructor(&ctor_name)?;
        let np = usize::from(ctor.num_params);
        let nf = usize::from(ctor.num_fields);
        if margs.len() != np + nf {
            return None;
        }
        let rule = rec.rec_rules.iter().find(|r| r.ctor_name == ctor_name)?;
        if usize::from(rule.ctor_telescope_size_wo_params) != nf
            || self.ctx.read_levels(levels).len() != self.ctx.read_levels(rec.info.uparams).len()
        {
            return None;
        }
        let rhs = self.ctx.subst_expr_levels(rule.val, rec.info.uparams, levels);
        let nprefix = usize::from(rec.num_params + rec.num_motives + rec.num_minors);
        let nargs = args[..nprefix]
            .iter()
            .chain(&margs[np..])
            .chain(&args[major_idx + 1..])
            .copied()
            .collect::<Vec<_>>();
        Some(self.ctx.foldl_apps(rhs, nargs.into_iter()))
    }

    /// Open the binder of `body` with a fresh level, returning the opened
    /// body and the level.
    fn fz_open(
        &mut self,
        binder_name: NamePtr<'t>,
        binder_style: crate::expr::BinderStyle,
        binder_type: ExprPtr<'t>,
        body: ExprPtr<'t>,
    ) -> (ExprPtr<'t>, u32) {
        let lvl = self.ctx.dbj_level_counter;
        let x = self.ctx.mk_dbj_level(binder_name, binder_style, binder_type);
        (self.ctx.inst(body, &[x]), lvl)
    }

    /// Fuse `body`, opened at level `lvl`, and close it again; `None` when
    /// fusion left it unchanged.
    fn fz_under(&mut self, st: &mut Fuser<'t>, opened: ExprPtr<'t>, lvl: u32) -> Option<ExprPtr<'t>> {
        let r = self.fz(st, opened);
        self.ctx.dbj_level_counter = lvl;
        (r != opened).then(|| self.ctx.abstr_levels_at(r, lvl, lvl + 1))
    }

    fn fz(&mut self, st: &mut Fuser<'t>, e: ExprPtr<'t>) -> ExprPtr<'t> {
        if st.overflow {
            return e;
        }
        let n = self.ctx.read_expr(e);
        if matches!(n, Var { .. } | Local { .. } | Sort { .. } | NatLit { .. } | StringLit { .. } | Pi { .. }) {
            return e;
        }
        if let Some(&r) = st.memo.get(&e) {
            return r;
        }
        if st.budget == 0 {
            st.overflow = true;
            return e;
        }
        st.budget -= 1;
        let r = match n {
            Lambda { binder_name, binder_style, binder_type, body, .. } => {
                let (opened, lvl) = self.fz_open(binder_name, binder_style, binder_type, body);
                match self.fz_under(st, opened, lvl) {
                    Some(b) => self.ctx.mk_lambda(binder_name, binder_style, binder_type, b),
                    None => e,
                }
            }
            Let { binder_name, binder_type, val, body, nondep, .. } => {
                let v = self.fz(st, val);
                let (opened, lvl) =
                    self.fz_open(binder_name, crate::expr::BinderStyle::Default, binder_type, body);
                match (self.fz_under(st, opened, lvl), v != val) {
                    (None, false) => e,
                    (b, _) => {
                        let b = b.unwrap_or(body);
                        self.ctx.mk_let(binder_name, binder_type, v, b, nondep)
                    }
                }
            }
            Proj { ty_name, idx, structure, .. } => {
                let s = self.fz(st, structure);
                match self.ctor_app_field(s, ty_name, idx) {
                    Some(f) => f,
                    None if s != structure => self.ctx.mk_proj(ty_name, idx, s),
                    None => e,
                }
            }
            App { .. } | Const { .. } => self.fz_app(st, e),
            _ => e,
        };
        st.memo.insert(e, r);
        r
    }

    fn fz_app(&mut self, st: &mut Fuser<'t>, e: ExprPtr<'t>) -> ExprPtr<'t> {
        let (mut f, mut args) = self.ctx.unfold_apps(e);
        for a in args.iter_mut() {
            *a = self.fz(st, *a);
        }
        let mut unfolds = 0;
        while !st.overflow {
            match self.ctx.read_expr(f) {
                Lambda { .. } if !args.is_empty() => {
                    let mut i = 0;
                    let mut b = f;
                    while i < args.len() {
                        let Lambda { body, .. } = self.ctx.read_expr(b) else { break };
                        b = body;
                        i += 1;
                    }
                    if !self.share_ok(st, b, &args[..i]) {
                        let f2 = self.fz(st, f);
                        if f2 == f {
                            break;
                        }
                        f = f2;
                        continue;
                    }
                    let t = self.ctx.inst(b, &args[..i]);
                    args.drain(..i);
                    if st.depth >= MAX_DEPTH {
                        f = t;
                        break;
                    }
                    st.depth += 1;
                    let t = self.fz(st, t);
                    st.depth -= 1;
                    if args.is_empty() {
                        return t;
                    }
                    let (h, mut targs) = self.ctx.unfold_apps(t);
                    targs.append(&mut args);
                    f = h;
                    args = targs;
                }
                Const { name, levels, .. } => {
                    if self.env.get_recursor(&name).is_some() {
                        let Some(t) = self.try_iota(name, levels, &args) else { break };
                        if st.depth >= MAX_DEPTH {
                            break;
                        }
                        st.depth += 1;
                        let t = self.fz(st, t);
                        st.depth -= 1;
                        return t;
                    }
                    if unfolds >= MAX_UNFOLDS || !self.fuse_unfoldable(name) {
                        break;
                    }
                    let Some((uparams, val)) = self.env.get_declar_val(&name) else { break };
                    if self.ctx.read_levels(levels).len() != self.ctx.read_levels(uparams).len() {
                        break;
                    }
                    unfolds += 1;
                    f = self.ctx.subst_expr_levels(val, uparams, levels);
                    if args.is_empty() {
                        return self.fz(st, f);
                    }
                }
                _ => {
                    let f2 = self.fz(st, f);
                    if f2 == f {
                        break;
                    }
                    match self.ctx.read_expr(f2) {
                        App { .. } => {
                            let (h, mut fargs) = self.ctx.unfold_apps(f2);
                            fargs.append(&mut args);
                            f = h;
                            args = fargs;
                        }
                        Lambda { .. } | Const { .. } => f = f2,
                        _ => {
                            f = f2;
                            break;
                        }
                    }
                }
            }
        }
        self.ctx.foldl_apps(f, args.into_iter())
    }
}

fn push_children<'t>(n: crate::expr::Expr<'t>, todo: &mut Vec<ExprPtr<'t>>) {
    match n {
        App { fun, arg, .. } => {
            todo.push(fun);
            todo.push(arg);
        }
        Lambda { binder_type, body, .. } | Pi { binder_type, body, .. } => {
            todo.push(binder_type);
            todo.push(body);
        }
        Let { binder_type, val, body, .. } => {
            todo.push(binder_type);
            todo.push(val);
            todo.push(body);
        }
        Proj { structure, .. } => todo.push(structure),
        _ => {}
    }
}
