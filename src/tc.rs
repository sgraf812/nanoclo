use crate::rapier_core::{
    clo_le, Clo, Entry, EnvId, SClo, SpineVec, ENV_NIL,
};
use crate::env::{Declar, DeclarInfo, Env, ReducibilityHint};
use crate::expr::Expr;
use crate::util::{
    nat_div, nat_gcd, nat_land, nat_lor, nat_mod, nat_sub, nat_xor,
    ExportFile, ExprPtr, LevelPtr, LevelsPtr, NamePtr, TcCtx,
};
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use std::error::Error;

#[derive(Clone, Copy, PartialEq, Eq)]
enum NatOp {
    Succ,
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

/// Conversion steps one speculative comparison may spend, counting everything
/// it nests. Chosen so that the comparisons which settle a pair on the
/// corpora finish inside it: raising it to 65536 costs `discarded-argument`
/// its bound, lowering it to 1024 costs `args-before-unfold` its answer.
const SPEC_BUDGET: u64 = 4096;

use Expr::*;
use InferFlag::*;



/// A flag that accompanies calls to type inference; if the flag is `Check`,
/// we perform additional definitional equality checks (for example, the type of an
/// argument to a lambda is the same type as the binder in the labmda). These checks
/// are costly however, and in some cases we're using inference during reduction of
/// expressions we know to be well-typed, so we can pass the flag `InferOnly` to omit
/// these checks when they are not needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum InferFlag {
    InferOnly,
    Check,
}

pub struct TypeChecker<'x, 't, 'p> {
    pub(crate) ctx: &'x mut TcCtx<'t, 'p>,
    /// An immutable reference to an environment, which contains declarations and notation.
    /// To accommodate the temporary declarations created while checking nested inductives,
    /// the environment may have a temporary extension which also holds declarations, and
    /// is searched before the persistent environment.
    ///
    /// This is stored as a field in `TypeChecker` rather than being placed in `TcCtx` so
    /// that the borrow checker will allow us to mutably reference `TcCtx` while we have
    /// outstanding references to environment declarations. Rust can tell that borrows
    /// of different struct fields are exclusive, but it can't analyze what fields of a given
    /// field's type are being exclusively borrowed.
    pub(crate) env: &'x Env<'x, 't>,
    /// The caches for things like inference, reduction, and equality checking.
    /// If this type checker is being used to check a simple declaration, this field will
    /// contain the universe parameters of that declaration. This is used in a couple of places
    /// to make sure that all of the universe paramters actually used in a declaration `d` are
    /// properly represented in the declaration's uparams info.
    pub(crate) declar_info: Option<DeclarInfo<'t>>,
}

impl<'p> ExportFile<'p> {
    /// The entry point for checking a declaration `d`, creating a fresh
    /// checking context, and with it a fresh expression dag: everything the
    /// check interns is released when `d` is done. Terms parsed from the
    /// export file live in the shared dag and are unaffected.
    pub fn check_declar(&self, d: &Declar<'p>) {
        self.with_ctx(|ctx| {
            self.check_declar_in(ctx, d);
            ctx.rp.flush_ctrs();
        })
    }

    /// Check a declaration in an existing context.
    pub fn check_declar_in<'t>(&'t self, ctx: &mut TcCtx<'t, 'p>, d: &Declar<'p>) {
        ctx.rp.reset_decl();
        use Declar::*;
        match d {
            Axiom { .. } => ctx.with_tc_and_declar(*d.info(), |tc| tc.check_declar_info(d).unwrap()),
            Inductive(..) => self.check_inductive_declar_in(ctx, d),
            Quot { .. } => crate::quot::check_quot(ctx, d),
            Definition { val, .. } | Theorem { val, .. } | Opaque { val, .. } =>
                ctx.with_tc_and_declar(*d.info(), |tc| {
                    tc.check_declar_info(d).unwrap();
                    let inferred_type = tc.infer(*val, crate::tc::InferFlag::Check);
                    tc.assert_def_eq(inferred_type, d.info().ty);
                }),
            Constructor(ctor_data) => {
                ctx.with_tc_and_declar(*d.info(), |tc| tc.check_declar_info(d).unwrap());
                assert!(self.declars.get(&ctor_data.inductive_name).is_some());
            }
            Recursor(recursor_data) => {
                ctx.with_tc_and_declar(*d.info(), |tc| tc.check_declar_info(d).unwrap());
                for ind_name in recursor_data.all_inductives.iter() {
                    assert!(self.declars.get(ind_name).is_some())
                }
            }
        }
    }

    /// Check all declarations in this export file using a single thread.
    /// Runs on a dedicated large-stack thread (the rapier core recurses over
    /// term structure). Each declaration gets its own context, so the
    /// expressions built while checking it are released when it finishes.
    pub(crate) fn check_all_declars_serial(&self) {
        std::thread::scope(|sco| {
            std::thread::Builder::new()
                .stack_size(crate::STACK_SIZE)
                .spawn_scoped(sco, || {
                    let thresh: u128 = std::env::var("RAPIER_DECLTIME")
                        .ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
                    let report = std::env::var("RAPIER_DECLTIME").is_ok();
                    let skip: usize = std::env::var("RAPIER_SKIP_UNTIL")
                        .ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                    let stop_after: Option<usize> = std::env::var("RAPIER_STOP_AFTER")
                        .ok().and_then(|v| v.parse().ok());
                    let repeat: usize = std::env::var("RAPIER_REPEAT")
                        .ok().and_then(|v| v.parse().ok()).unwrap_or(1);
                    for (i, declar) in self.declars.values().enumerate() {
                        if i < skip {
                            continue
                        }
                        if let Some(stop) = stop_after {
                            if i > stop {
                                break
                            }
                        }
                        for _ in 1..repeat {
                            self.check_declar(declar);
                        }
                        if report {
                            if thresh == 0 {
                                self.with_ctx(|ctx| eprintln!(
                                    "ENTER\t{}\t{:?}", i, ctx.debug_print(declar.info().name)));
                            }
                            let t0 = std::time::Instant::now();
                            self.check_declar(declar);
                            let us = t0.elapsed().as_micros();
                            if us >= thresh {
                                self.with_ctx(|ctx| {
                                    eprintln!("DECL\t{}\t{}\t{:?}", i, us, ctx.debug_print(declar.info().name))
                                });
                            }
                        } else {
                            self.check_declar(declar);
                        }
                    }
                })
                .unwrap()
                .join()
                .expect("serial check thread panicked while being joined");
        });
    }

    /// Check all declarations in this export file, spawning `num_threads` as
    /// checkers, each with one context reused for all of its declarations.
    fn check_all_declars_par(&self, num_threads: usize) {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
        use std::thread;
        let task_num = AtomicUsize::new(0);
        thread::scope(|sco| {
            let mut handles = Vec::new();
            for i in 0..num_threads {
                handles.push(
                    thread::Builder::new()
                        .name(format!("thread_{}", i))
                        .stack_size(crate::STACK_SIZE)
                        .spawn_scoped(sco, || loop {
                            let idx = task_num.fetch_add(1, Relaxed);
                            if let Some((_, declar)) = self.declars.get_index(idx) {
                                self.check_declar(declar);
                            } else {
                                break
                            }
                        })
                        .unwrap(),
                )
            }
            for t in handles {
                t.join().expect("A thread in `check_all_declars` panicked while being joined");
            }
        });
    }

    /// Check all of the declarations in this export file on the specified number
    /// of threads (checking will be serial on the main thread is num_threads <= 1).
    pub fn check_all_declars(&self) {
        if self.config.num_threads > 1 {
            self.check_all_declars_par(self.config.num_threads)
        } else {
            self.check_all_declars_serial()
        }
    }
}

impl<'x, 't: 'x, 'p: 't> TypeChecker<'x, 't, 'p> {
    pub fn new(dag: &'x mut TcCtx<'t, 'p>, env: &'x Env<'x, 't>, declar_info: Option<DeclarInfo<'t>>) -> Self {
        assert_eq!(dag.dbj_level_counter, 0);
        Self { ctx: dag, env, declar_info }
    }

    /// Conduct the preliminary checks done on all declarations; a declaration
    /// must not contain duplicate universe parameters, mut not have free variables,
    /// and must have an ascribed type that is actually a type (`infer declaration.type` must
    /// be a sort).
    pub(crate) fn check_declar_info(&mut self, d: &Declar<'t>) -> Result<(), Box<dyn Error>> {
        let info = d.info();
        assert!(self.ctx.no_dupes_all_params(info.uparams));
        assert!(!self.ctx.has_fvars(info.ty));
        let inferred_type = self.infer(info.ty, Check);
        let sort = self.ensure_sort(inferred_type);

        // This is sort of a "soft" check in terms of soundness, but for theorems, ensure 
        // that they're propositions.
        if let Declar::Theorem {..} = d {
            if !self.ctx.is_zero(sort) {
                return Err(Box::<dyn Error>::from(format!("Theorem type for {:?} must be `Prop` (sort 0); found type {:?}",
                    self.ctx.debug_print(info.name),
                    self.ctx.debug_print(sort)
                )))
            }
        } 
        Ok(())
    }




    pub(crate) fn ensure_infers_as_sort(&mut self, e: ExprPtr<'t>) -> LevelPtr<'t> {
        let infd = self.infer(e, Check);
        self.ensure_sort(infd)
    }

    pub(crate) fn ensure_sort(&mut self, e: ExprPtr<'t>) -> LevelPtr<'t> {
        if let Sort { level, .. } = self.ctx.read_expr(e) {
            return level
        }
        let whnfd = self.whnf(e);
        match self.ctx.read_expr(whnfd) {
            Sort { level, .. } => level,
            _ => panic!("ensur_sort could not produce a sort"),
        }
    }








    // For structures that carry no additional information, elements with the same type are def_eq.

    


    pub(crate) fn infer_then_whnf(&mut self, e: ExprPtr<'t>, flag: InferFlag) -> ExprPtr<'t> {
        let ty = self.infer(e, flag);
        self.whnf(ty)
    }


    /// Delegates to the rapier delayed-instantiation core (`rapier_core.rs`).
    pub(crate) fn infer(&mut self, e: ExprPtr<'t>, flag: InferFlag) -> ExprPtr<'t> {
        self.rp_infer(crate::rapier_core::Clo::of(e), flag)
    }




    //fn infer_app(&mut self, e: ExprPtr<'t>, flag: InferFlag) -> ExprPtr<'t> {
    //    match self.ctx.read_expr(e) {
    //        App {fun, arg, ..} => {
    //            let fun_ty = self.infer_then_whnf(fun, flag);
    //            match self.ctx.read_expr(fun_ty) {
    //                Pi {binder_type, body, ..} => {
    //                    if flag == InferFlag::Check {
    //                        let arg_ty = self.infer(arg, flag);
    //                        let outer_scope_eager_setting = self.ctx.eager_mode;
    //                        if self.ctx.is_eager_reduce_app(arg) {
    //                            self.ctx.eager_mode = true;
    //                        }
    //                        self.assert_def_eq(binder_type, arg_ty);
    //                        self.ctx.eager_mode = outer_scope_eager_setting;
    //                    }
    //                    self.ctx.inst(body, &[arg])
    //                },
    //                _ => panic!()
    //            }
    //        },
    //        _ => panic!()
    //    }
    //}



    
    // Not well tested, used for introspection/debugging.

    /// Delegates to the rapier delayed-instantiation core (`rapier_core.rs`).
    pub fn whnf(&mut self, e: ExprPtr<'t>) -> ExprPtr<'t> {
        let s = self.rp_whnf_clo(crate::rapier_core::Clo::of(e));
        let out = self.rp_sclo_to_expr(&s);
        // mirror upstream whnf: sort levels come out simplified
        if let Sort { level, .. } = self.ctx.read_expr(out) {
            let level = self.ctx.simplify(level);
            return self.ctx.mk_sort(level)
        }
        out
    }











    pub fn assert_def_eq(&mut self, u: ExprPtr<'t>, v: ExprPtr<'t>) { assert!(self.def_eq(u, v)) }

    /// Delegates to the rapier delayed-instantiation core (`rapier_core.rs`).
    pub fn def_eq(&mut self, x: ExprPtr<'t>, y: ExprPtr<'t>) -> bool {
        self.rp_is_def_eq(crate::rapier_core::Clo::of(x), crate::rapier_core::Clo::of(y))
    }





    

    pub fn reduce_quot(&mut self, c_name: NamePtr<'t>, args: &[ExprPtr<'t>]) -> Option<ExprPtr<'t>> {
        if !matches!(self.env.get_declar(&c_name), Some(Declar::Quot {..})) {
            return None
        }
        let (qmk, rest_idx) = if c_name == self.ctx.export_file.name_cache.quot_lift? {
            let qmk = args.get(5).copied()?;
            (self.whnf(qmk), 6)
        } else if c_name == self.ctx.export_file.name_cache.quot_ind? {
            let qmk = args.get(4).copied()?;
            (self.whnf(qmk), 5)
        } else {
            return None
        };
        {
            let (qmk_const, qmk_args) = self.ctx.unfold_apps(qmk);
            match self.ctx.read_expr(qmk_const) {
                Const { name, .. } if name == self.ctx.export_file.name_cache.quot_mk? && qmk_args.len() == 3 => (),
                _ => return None,
            };
        }
        let f = args.get(3).copied()?;
        let appd = match self.ctx.read_expr(qmk) {
            App { arg, .. } => self.ctx.mk_app(f, arg),
            _ => panic!("Quot iota"),
        };
        Some(self.ctx.foldl_apps(appd, args.iter().copied().skip(rest_idx)))
    }

    // We only need the name and reducibility from this.











    pub fn is_prop(&mut self, e: ExprPtr<'t>) -> (bool, ExprPtr<'t>) {
        let ty = self.infer_then_whnf(e, InferOnly);
        match self.ctx.read_expr(ty) {
            Sort { level, .. } => (self.ctx.is_zero(level), ty),
            _ => (false, ty),
        }
    }

    pub fn may_be_prop(&mut self, e: ExprPtr<'t>) -> (bool, ExprPtr<'t>) {
        let ty = self.infer_then_whnf(e, InferOnly);
        match self.ctx.read_expr(ty) {
            Sort { level, .. } => (self.ctx.may_be_prop(level), ty),
            _ => (false, ty),
        }
    }

    pub fn is_proof(&mut self, e: ExprPtr<'t>) -> (bool, ExprPtr<'t>) {
        let infd = self.infer(e, InferOnly);
        (self.is_prop(infd).0, infd)
    }

    // ---- whnf ----

    pub(crate) fn rp_mk_sclo(&self, c: Clo<'t>) -> SClo<'t> {
        let mut spine = SpineVec::new();
        let mut e = c.e;
        while let App { fun, arg, .. } = self.ctx.read_expr(e) {
            spine.push(Clo { e: arg, env: c.env });
            e = fun;
        }
        spine.reverse();
        SClo { head: Clo { e, env: c.env }, spine }
    }

    /// whnf recurses through reduce_nat and nat_lit back into itself, so the
    /// depth follows the term's reduction rather than its size. Grow the
    /// stack on demand here rather than sizing a thread stack for the worst
    /// input.
    pub(crate) fn rp_whnf_clo(&mut self, c: Clo<'t>) -> SClo<'t> {
        let d = self.rp_probe_suspend();
        let r = stacker::maybe_grow(256 * 1024, 16 * 1024 * 1024, || self.rp_whnf_clo_inner(c));
        self.rp_probe_resume(d);
        r
    }

    fn rp_whnf_clo_inner(&mut self, c: Clo<'t>) -> SClo<'t> {
        let c = self.rp_norm_clo(c);
        match self.ctx.read_expr(c.e) {
            NatLit { .. } | StringLit { .. } | Sort { .. } | Pi { .. } | Lambda { .. }
            | Local { .. } => return SClo { head: c, spine: SpineVec::new() },
            _ => {}
        }
        let k = self.rp_key(c);
        if let Some(r) = self.ctx.rp.whnf_cache.get(&k) {
            let r = r.clone();
            self.ctx.rp.ctrs[4] += 1;
            return r;
        }
        self.ctx.rp.ctrs[5] += 1;
        self.ctx.rp.ctrs[2] += 1;
        let t = self.rp_whnf_core_clo(c);
        let r = self.rp_whnf_loop(t);
        // whnf_cache is never evicted: dropping a weak-head normal form makes
        // the checker redo whole reduction sequences, which on nested redexes
        // is the difference between linear and exponential. Measured on the
        // nested-beta family at n=1500: ~1s with it intact, >90s when capped,
        // while capping the other memos costs nothing even at 2^10.
        self.ctx.rp.whnf_cache.insert(k, r.clone());
        r
    }

    pub(crate) fn rp_whnf(&mut self, s: SClo<'t>) -> SClo<'t> {
        self.ctx.rp.ctrs[2] += 1;
        let t = self.rp_whnf_core(s);
        self.rp_whnf_loop(t)
    }

    fn rp_whnf_loop(&mut self, mut t: SClo<'t>) -> SClo<'t> {
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

    /// `whnf_core` entered at a closure: the head and spine come from `c.e`,
    /// and a head already in weak head normal form needs no spine.
    fn rp_whnf_core_clo(&mut self, c: Clo<'t>) -> SClo<'t> {
        let c = self.rp_norm_clo(c);
        match self.ctx.read_expr(c.e) {
            NatLit { .. } | StringLit { .. } | Sort { .. } | Pi { .. } | Lambda { .. }
            | Local { .. } => return SClo { head: c, spine: SpineVec::new() },
            _ => {}
        }
        let s = self.rp_mk_sclo(c);
        self.rp_whnf_core_ext(s, false, false)
    }

    /// `cheap`: skip recursor and projection reduction (the C++ kernel's
    /// cheap_rec/cheap_proj), so def-eq can compare stuck same-head
    /// applications structurally before forcing e.g. Nat.below towers.
    fn rp_whnf_core_ext(&mut self, s: SClo<'t>, cheap_rec: bool, cheap_proj: bool) -> SClo<'t> {
        self.ctx.rp.ctrs[1] += 1;
        let SClo { head, spine } = s;
        let mut e = head.e;
        let mut env = head.env;
        // reversed spine: last element is the innermost (next) argument
        let mut rsp: SpineVec<'t> = spine.into_iter().rev().collect();
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
        let r = if self.ctx.read_levels(levels).is_empty() {
            def_value
        } else if !self.env.has_temp_ext() {
            // the head Const expr is closed and fvar-free; the instantiation
            // depends only on it, so it can be cached per thread
            if let Some(&r) = self.ctx.rp.g_unfold.get(&s.head.e) {
                self.ctx.rp.ctrs[8] += 1;
                r
            } else {
                self.ctx.rp.ctrs[9] += 1;
                let r = self.ctx.subst_expr_levels(def_value, def_uparams, levels);
                self.ctx.rp.g_unfold.insert(s.head.e, r);
                r
            }
        } else {
            self.ctx.subst_expr_levels(def_value, def_uparams, levels)
        };
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
        SClo { head: Clo::of(e), spine: SpineVec::new() }
    }

    fn rp_mk_bool_sclo(&mut self, b: bool) -> Option<SClo<'t>> {
        let e = self.ctx.bool_to_expr(b)?;
        Some(SClo { head: Clo::of(e), spine: SpineVec::new() })
    }

    /// The Nat primitive this spined application would dispatch to, if any.
    /// Shared by `rp_reduce_nat` and lazy-delta's cheap pre-test.
    fn rp_nat_op(&self, s: &SClo<'t>) -> Option<NatOp> {
        let f = match self.ctx.read_expr(s.head.e) {
            Const { name, levels, .. } if self.ctx.read_levels(levels).is_empty() => Some(name),
            _ => return None,
        };
        use NatOp::*;
        let nc = self.ctx.export_file.name_cache;
        let (op, arity) = if f == nc.nat_succ {
            (Succ, 1)
        } else if f == nc.nat_add {
            (Add, 2)
        } else if f == nc.nat_sub {
            (Sub, 2)
        } else if f == nc.nat_mul {
            (Mul, 2)
        } else if f == nc.nat_pow {
            (Pow, 2)
        } else if f == nc.nat_gcd {
            (Gcd, 2)
        } else if f == nc.nat_mod {
            (Mod, 2)
        } else if f == nc.nat_div {
            (Div, 2)
        } else if f == nc.nat_beq {
            (Beq, 2)
        } else if f == nc.nat_ble {
            (Ble, 2)
        } else if f == nc.nat_land {
            (LAnd, 2)
        } else if f == nc.nat_lor {
            (LOr, 2)
        } else if f == nc.nat_xor {
            (XOr, 2)
        } else if f == nc.nat_shl {
            (Shl, 2)
        } else if f == nc.nat_shr {
            (Shr, 2)
        } else {
            return None;
        };
        (s.spine.len() == arity).then_some(op)
    }

    fn rp_reduce_nat(&mut self, s: &SClo<'t>) -> Option<SClo<'t>> {
        if !self.ctx.export_file.config.nat_extension {
            return None;
        }
        use NatOp::*;
        let op = self.rp_nat_op(s)?;
        if op == Succ {
            let v = self.rp_nat_lit(s.spine[0])?;
            return Some(self.rp_mk_nat_sclo(v + 1u32));
        }
        // exponent/shift bound is checked before the base is forced
        let b = self.rp_nat_lit(s.spine[1])?;
        if matches!(op, Pow | Shl) && b > BigUint::from(1u32 << 24) {
            return None;
        }
        let a = self.rp_nat_lit(s.spine[0])?;
        let r = match op {
            Succ => unreachable!(),
            Beq => return self.rp_mk_bool_sclo(a == b),
            Ble => return self.rp_mk_bool_sclo(a <= b),
            Add => a + b,
            Sub => nat_sub(a, b),
            Mul => a * b,
            Pow => num_traits::pow::Pow::pow(a, b.to_u32().unwrap()),
            Gcd => nat_gcd(&a, &b),
            Mod => nat_mod(a, b),
            Div => nat_div(a, b),
            LAnd => nat_land(a, b),
            LOr => nat_lor(a, b),
            XOr => nat_xor(&a, &b),
            Shl => a << b.to_u64().unwrap(),
            Shr => match b.to_u64() {
                Some(sh) => a >> sh,
                None => BigUint::zero(),
            },
        };
        Some(self.rp_mk_nat_sclo(r))
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
        // K conversion runs on the unevaluated major: it inspects only the
        // major's type, so a K-recursor's proof argument is replaced by the
        // nullary constructor without ever being reduced. The major itself is
        // whnf'd only afterwards (trivial when the conversion fired).
        let mut mj = if cheap_rec {
            let s0 = self.rp_mk_sclo(idx(major_idx));
            self.rp_whnf_core_ext(s0, cheap_rec, cheap_proj)
        } else if is_k && major_induct.is_some() {
            let s0 = self.rp_mk_sclo(idx(major_idx));
            let conv = self.rp_to_ctor_when_k(num_params, major_induct.unwrap(), s0.clone());
            if conv.head == s0.head && conv.spine == s0.spine {
                self.rp_whnf_clo(idx(major_idx))
            } else {
                self.rp_whnf(conv)
            }
        } else {
            self.rp_whnf_clo(idx(major_idx))
        };
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
            spine: app_type.spine[..(num_params as usize).min(app_type.spine.len())].iter().copied().collect(),
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
        // A structure whose universe an instantiation may send to zero is
        // left alone: expanding a proof into its fields would equate proofs
        // that proof irrelevance already equates on other grounds.
        let tyty = self.rp_infer_s(&e_type, InferOnly);
        let tyty_w = self.rp_whnf_clo(Clo::of(tyty));
        if let Sort { level, .. } = self.ctx.read_expr(tyty_w.head.e) {
            if tyty_w.spine.is_empty() && self.ctx.may_be_prop(level) {
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
        let mut spine: SpineVec<'t> =
            e_type.spine[..(num_params as usize).min(e_type.spine.len())].iter().copied().collect();
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

    // ---- bounded speculation ----
    //
    // Several points in conversion admit more than one next step, and taking
    // the wrong one can cost unboundedly more than the answer is worth.
    // Comparing the arguments of two applications of one constant is such a
    // point: the arguments may differ while the applications are equal, so a
    // comparison that runs long is evidence that unfolding the constant is
    // the cheaper route. Running it under a fuel bound turns "wait for the
    // answer" into "give it a fixed budget, then take the other route".
    //
    // A comparison abandoned this way has produced no answer, so nothing it
    // computed may be recorded as one. Reduction is left unbounded, since a
    // truncated weak head normal form would be recorded by the memos that
    // reduction shares with inference.

    fn rp_probe_enter(&mut self, budget: u64) -> (u64, bool, usize, usize) {
        let saved = (
            self.ctx.rp.probe_fuel,
            self.ctx.rp.probe_aborted,
            self.ctx.rp.probe_neg_log.len(),
            self.ctx.rp.probe_gneg.len(),
        );
        let outer = self.ctx.rp.probe_depth == 0;
        self.ctx.rp.probe_depth += 1;
        if outer {
            self.ctx.rp.probe_fuel = budget;
            self.ctx.rp.probe_aborted = false;
        }
        saved
    }

    /// Whether the comparison ran out of fuel, in which case its answer says
    /// nothing. A comparison that finished contributes what it found unequal.
    fn rp_probe_exit(&mut self, saved: (u64, bool, usize, usize)) -> bool {
        let aborted = self.ctx.rp.probe_aborted;
        self.ctx.rp.probe_depth -= 1;
        if self.ctx.rp.probe_depth == 0 {
            self.ctx.rp.probe_fuel = saved.0;
            self.ctx.rp.probe_aborted = saved.1;
        }
        if aborted {
            while self.ctx.rp.probe_neg_log.len() > saved.2 {
                let pk = self.ctx.rp.probe_neg_log.pop().unwrap();
                self.ctx.rp.probe_neg.remove(&pk);
            }
            self.ctx.rp.probe_gneg.truncate(saved.3);
        } else if self.ctx.rp.probe_depth == 0 {
            while let Some(pk) = self.ctx.rp.probe_neg_log.pop() {
                self.ctx.rp.eq_neg.insert(pk);
            }
            self.ctx.rp.probe_neg.clear();
            while let Some(gpk) = self.ctx.rp.probe_gneg.pop() {
                self.ctx.rp.g_eq_neg.insert(gpk);
            }
        }
        aborted
    }

    #[inline]
    fn rp_spend(&mut self) {
        if self.ctx.rp.probe_depth > 0 && !self.ctx.rp.probe_aborted {
            if self.ctx.rp.probe_fuel == 0 {
                self.ctx.rp.probe_aborted = true;
            } else {
                self.ctx.rp.probe_fuel -= 1;
            }
        }
    }

    #[inline]
    fn rp_in_probe(&self) -> bool { self.ctx.rp.probe_depth > 0 }

    /// Step outside the budget. Reduction and inference write memos that
    /// conversion shares with the rest of the checker, and an entry recorded
    /// from a truncated reduction would be read later as a final answer, so
    /// they run to completion whatever the budget says.
    #[inline]
    fn rp_probe_suspend(&mut self) -> u32 {
        std::mem::replace(&mut self.ctx.rp.probe_depth, 0)
    }

    #[inline]
    fn rp_probe_resume(&mut self, depth: u32) { self.ctx.rp.probe_depth = depth; }

    // ---- def-eq ----

    pub(crate) fn rp_is_def_eq(&mut self, t: Clo<'t>, s: Clo<'t>) -> bool {
        self.ctx.rp.ctrs[3] += 1;
        if self.rp_clo_eq(t, s) {
            return true;
        }
        self.rp_spend();
        if self.ctx.rp.probe_aborted {
            return false;
        }
        let (tk, sk) = (self.rp_norm_clo(t), self.rp_norm_clo(s));
        let (tkey, skey) = (self.rp_key(tk), self.rp_key(sk));
        if self.ctx.rp.eq_pos.known_eq(&tkey, &skey) {
            self.ctx.rp.ctrs[6] += 1;
            return true;
        }
        if let (Some(a), Some(b)) = (self.rp_global_key(tk), self.rp_global_key(sk)) {
            let gpk = if a.get_hash() <= b.get_hash() { (a, b) } else { (b, a) };
            if self.ctx.rp.g_eq_pos.known_eq(&a, &b) {
                self.ctx.rp.ctrs[6] += 1;
                return true;
            }
            if self.ctx.rp.g_eq_neg.contains(&gpk) {
                self.ctx.rp.ctrs[6] += 1;
                return false;
            }
            self.ctx.rp.ctrs[7] += 1;
            let r = self.rp_is_def_eq_clo(tk, sk);
            if r {
                self.ctx.rp.g_eq_pos.union(a, b);
            } else if self.rp_in_probe() {
                self.ctx.rp.probe_gneg.push(gpk);
            } else {
                self.ctx.rp.g_eq_neg.insert(gpk);
            }
            return r;
        }
        let pk = if clo_le(&tkey, &skey) { (tkey, skey) } else { (skey, tkey) };
        if self.ctx.rp.eq_neg.contains(&pk)
            || (self.rp_in_probe() && self.ctx.rp.probe_neg.contains(&pk))
        {
            self.ctx.rp.ctrs[6] += 1;
            return false;
        }
        self.ctx.rp.ctrs[7] += 1;
        let r = self.rp_is_def_eq_clo(tk, sk);
        if r {
            self.ctx.rp.eq_pos.union(tkey, skey);
        } else if self.rp_in_probe() {
            if self.ctx.rp.probe_neg.insert(pk) {
                self.ctx.rp.probe_neg_log.push(pk);
            }
        } else {
            self.ctx.rp.eq_neg.insert(pk);
        }
        r
    }

    /// Both sides enter `whnf_core` at a closure key, so the memo applies.
    fn rp_is_def_eq_clo(&mut self, t: Clo<'t>, s: Clo<'t>) -> bool {
        let tn = self.rp_whnf_core_clo(t);
        let sn = self.rp_whnf_core_clo(s);
        if self.rp_s_quick_eq(&tn, &sn) {
            return true;
        }
        self.rp_is_def_eq_s_core(tn, sn)
    }

    fn rp_is_def_eq_s(&mut self, t: SClo<'t>, s: SClo<'t>) -> bool {
        let tn = self.rp_whnf_core_ext(t, false, false);
        let sn = self.rp_whnf_core_ext(s, false, false);
        if self.rp_s_quick_eq(&tn, &sn) {
            return true;
        }
        self.rp_is_def_eq_s_core(tn, sn)
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
                        Proj { ty_name: n1, idx: i1, structure: e1, .. },
                        Proj { ty_name: n2, idx: i2, structure: e2, .. },
                    ) => {
                        if n1 == n2
                            && i1 == i2
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
        l1 == l2 || self.ctx.eq_antisymm(l1, l2)
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
        let level = self.rp_next_level(t.e, t.env).max(self.rp_next_level(s.e, s.env));
        let fv = self.rp_fvar_at(level, d);
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
        let n = self.ctx.read_expr(e);
        if n.has_fvars() {
            return true;
        }
        if n.num_loose_bvars() <= offset || env == ENV_NIL {
            return false;
        }
        let key = (e, env, offset);
        if let Some(r) = self.ctx.rp.clo_fvar_cache.get(&key) {
            return r;
        }
        let r = match n {
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
        self.ctx.rp.clo_fvar_cache.insert(key, r);
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
        self.rp_spend();
        if self.ctx.rp.probe_aborted {
            return Err((tn, sn));
        }
        if self.rp_s_quick_eq(&tn, &sn) {
            return Ok(true);
        }
        if let Some(b) = self.rp_def_eq_offset(&tn, &sn) {
            return Ok(b);
        }
        if (self.rp_nat_op(&tn).is_some() || self.rp_nat_op(&sn).is_some())
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
                        if same_const {
                            let saved = self.rp_probe_enter(SPEC_BUDGET);
                            let r = self.rp_def_eq_spines(&tn, &sn);
                            let aborted = self.rp_probe_exit(saved);
                            if r && !aborted {
                                return Ok(true);
                            }
                            if aborted {
                                self.ctx.rp.ctrs[24] += 1;
                            } else {
                                self.ctx.rp.ctrs[23] += 1;
                            }
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
        let level = self
            .rp_next_level(t.head.e, t.head.env)
            .max(self.rp_next_level(s.head.e, s.head.env))
            .max(self.rp_max_level(d));
        let fv = self.rp_fvar_at(level, d);
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

    /// Could `ty` be a proposition? A universe parameter stands for a level
    /// that an instantiation may send to zero, so it counts.
    fn rp_may_be_prop(&mut self, ty: ExprPtr<'t>) -> bool {
        let sort = self.rp_infer(Clo::of(ty), InferOnly);
        let w = self.rp_whnf_clo(Clo::of(sort));
        w.spine.is_empty()
            && match self.ctx.read_expr(w.head.e) {
                Sort { level, .. } => self.ctx.may_be_prop(level),
                _ => false,
            }
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
        let d = self.rp_probe_suspend();
        let r = self.rp_infer_go(c, flag);
        self.rp_probe_resume(d);
        r
    }

    fn rp_infer_go(&mut self, c: Clo<'t>, flag: InferFlag) -> ExprPtr<'t> {
        self.ctx.rp.ctrs[0] += 1;
        let n = self.ctx.read_expr(c.e);
        match n {
            Var { dbj_idx, .. } => match self.rp_lookup(c.env, dbj_idx) {
                Entry::Neu(fv) => {
                    self.ctx.rp.ctrs[19] += 1;
                    self.rp_fvar_type(fv)
                }
                Entry::Val(e2, env2) => {
                    self.ctx.rp.ctrs[18] += 1;
                    self.rp_infer(Clo { e: e2, env: env2 }, flag)
                }
            },
            Local { binder_type, .. } => {
                self.ctx.rp.ctrs[20] += 1;
                binder_type
            }
            Sort { level, .. } => {
                self.ctx.rp.ctrs[21] += 1;
                if flag == Check {
                    self.rp_check_level(level);
                }
                let l2 = self.ctx.succ(level);
                self.ctx.mk_sort(l2)
            }
            Const { name, levels, .. } => {
                self.ctx.rp.ctrs[22] += 1;
                if !self.env.has_temp_ext() {
                    if let Some(&r) = self.ctx.rp.g_inst_ty.get(&c.e) {
                        if flag == Check {
                            // checks still run per occurrence, instantiation reused
                            for l in self.ctx.read_levels(levels).iter().copied() {
                                self.rp_check_level(l);
                            }
                        }
                        return r;
                    }
                    let r = self.rp_infer_const(name, levels, flag);
                    self.ctx.rp.g_inst_ty.insert(c.e, r);
                    return r;
                }
                self.rp_infer_const(name, levels, flag)
            }
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
                    Check => self.ctx.rp.infer_cache_check.get(&key),
                    InferOnly => self.ctx.rp.infer_cache_only.get(&key),
                };
                if let Some(&r) = cached {
                    self.ctx.rp.ctrs[16] += 1;
                    return r;
                }
                self.ctx.rp.ctrs[17] += 1;
                let r = match n {
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
                    Check => self.ctx.rp.infer_cache_check.insert(key, r),
                    InferOnly => self.ctx.rp.infer_cache_only.insert(key, r),
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
        // The whole run of binders is opened before the body is inferred, so
        // the fvars standing for them are abstracted out of the inferred type
        // in one traversal rather than one traversal per binder.
        let mut env = c.env;
        let mut e = c.e;
        let mut binders = Vec::new();
        let mut start = 0u32;
        while let Lambda { binder_name, binder_style, binder_type, body, .. } =
            self.ctx.read_expr(e)
        {
            let d = self.rp_reify(Clo { e: binder_type, env });
            if flag == Check {
                let dty = self.rp_infer(Clo::of(d), flag);
                self.rp_ensure_sort(dty);
            }
            let level = self.rp_next_level(e, env).max(self.rp_max_level(d));
            if binders.is_empty() {
                start = level;
            }
            let fv = self.rp_fvar_at(level, d);
            env = self.rp_push_entry(env, Entry::Neu(fv));
            binders.push((binder_name, binder_style, d));
            e = body;
        }
        let bt = self.rp_infer(Clo { e, env }, flag);
        let bt = self.rp_cheap_beta_reduce(bt);
        let n = u32::try_from(binders.len()).unwrap();
        let mut r = self.ctx.abstr_levels_at(bt, start, start + n);
        for (i, (binder_name, binder_style, d)) in binders.into_iter().enumerate().rev() {
            let i = u32::try_from(i).unwrap();
            let d = self.ctx.abstr_levels_at(d, start, start + i);
            r = self.ctx.mk_pi(binder_name, binder_style, d, r);
        }
        r
    }

    fn rp_infer_pi(&mut self, c: Clo<'t>, flag: InferFlag) -> ExprPtr<'t> {
        let Pi { binder_type, body, .. } = self.ctx.read_expr(c.e) else {
            unreachable!()
        };
        let d = self.rp_reify(Clo { e: binder_type, env: c.env });
        let dty = self.rp_infer(Clo::of(d), flag);
        let u = self.rp_ensure_sort(dty);
        let level = self.rp_next_level(c.e, c.env).max(self.rp_max_level(d));
        let fv = self.rp_fvar_at(level, d);
        let env2 = self.rp_push_entry(c.env, Entry::Neu(fv));
        let bt = self.rp_infer(Clo { e: body, env: env2 }, flag);
        let s = self.rp_ensure_sort(bt);
        // mkLevelIMax': imax with immediate simplifications
        let lvl = self.ctx.imax(u, s);
        let lvl = self.ctx.simplify(lvl);
        self.ctx.mk_sort(lvl)
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
        let mut f_ty: Clo<'t> = Clo::of(self.rp_infer(s.head, flag));
        for &arg in s.spine.iter() {
            let fw = if matches!(self.ctx.read_expr(f_ty.e), Pi { .. }) {
                SClo { head: f_ty, spine: SpineVec::new() }
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
        let is_prop_ty = self.rp_may_be_prop(s_ty);
        for fi in 0..idx {
            let rw = self.rp_whnf_clo(r);
            let Pi { binder_type: dom, body, .. } = self.ctx.read_expr(rw.head.e) else {
                panic!("invalid projection");
            };
            assert!(rw.spine.is_empty(), "invalid projection");
            if self.lbr(body) > 0 && is_prop_ty {
                let d = self.rp_reify(Clo { e: dom, env: rw.head.env });
                assert!(self.rp_is_prop(d), "infer_proj prop");
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
            assert!(self.rp_is_prop(d), "infer_proj prop");
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
