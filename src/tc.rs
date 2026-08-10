use crate::closure::{
    Clo, Entry, SClo, SpineVec, ENV_NIL,
};
use crate::env::{Declar, DeclarInfo, Env};
use crate::expr::Expr;
use crate::util::{
    ExportFile, ExprPtr, LevelPtr, LevelsPtr, NamePtr, TcCtx,
};
use num_traits::Zero;
use std::error::Error;

/// Conversion steps one speculative comparison may spend, counting everything
/// it nests. Chosen so that the comparisons which settle a pair on the
/// corpora finish inside it: raising it to 65536 costs `discarded-argument`
/// its bound, lowering it to 1024 costs `args-before-unfold` its answer.
pub(crate) const SPEC_BUDGET: u64 = 4096;


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
        ctx.nb.reset_decl();
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
    /// Runs on a dedicated large-stack thread (conversion recurses over
    /// term structure). Each declaration gets its own context, so the
    /// expressions built while checking it are released when it finishes.
    pub(crate) fn check_all_declars_serial(&self) {
        std::thread::scope(|sco| {
            std::thread::Builder::new()
                .stack_size(crate::STACK_SIZE)
                .spawn_scoped(sco, || {
                    let thresh: u128 = std::env::var("NANOCLO_DECLTIME")
                        .ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
                    let report = std::env::var("NANOCLO_DECLTIME").is_ok();
                    let skip: usize = std::env::var("NANOCLO_SKIP_UNTIL")
                        .ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                    let stop_after: Option<usize> = std::env::var("NANOCLO_STOP_AFTER")
                        .ok().and_then(|v| v.parse().ok());
                    let repeat: usize = std::env::var("NANOCLO_REPEAT")
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

    /// `ensure_sort` without reifying the reduced type.
    fn ensure_sort_clo(&mut self, ty: ExprPtr<'t>) -> LevelPtr<'t> {
        if let Sort { level, .. } = self.ctx.read_expr(ty) {
            return level;
        }
        let v = self.nb_of_clo(Clo::of(ty));
        let v = self.nb_force(0, v);
        let v = self.nb_whnf(0, v);
        if let crate::nbe::Value::Sort { level } = self.ctx.nb.get(v) {
            return level;
        }
        panic!("ensur_sort could not produce a sort")
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


    /// Delegates to the delayed-instantiation core (`closure.rs`).
    pub(crate) fn infer(&mut self, e: ExprPtr<'t>, flag: InferFlag) -> ExprPtr<'t> {
        self.infer_clo(crate::closure::Clo::of(e), flag)
    }




    pub fn whnf(&mut self, e: ExprPtr<'t>) -> ExprPtr<'t> {
        let v = self.nb_of_clo(crate::closure::Clo::of(e));
        let v = self.nb_force(0, v);
        let v = self.nb_whnf(0, v);
        let out = self.nb_readback(v);
        // mirror upstream whnf: sort levels come out simplified
        if let Sort { level, .. } = self.ctx.read_expr(out) {
            let level = self.ctx.simplify(level);
            return self.ctx.mk_sort(level)
        }
        out
    }











    pub fn assert_def_eq(&mut self, u: ExprPtr<'t>, v: ExprPtr<'t>) { assert!(self.def_eq(u, v)) }

    /// Delegates to the delayed-instantiation core (`closure.rs`).
    pub fn def_eq(&mut self, x: ExprPtr<'t>, y: ExprPtr<'t>) -> bool {
        self.is_def_eq(crate::closure::Clo::of(x), crate::closure::Clo::of(y))
    }





    


    // We only need the name and reducibility from this.












    fn is_prop_of(&mut self, e: ExprPtr<'t>) -> bool {
        if let Some(&b) = self.ctx.rp.prop_cache.get(&e) {
            return b;
        }
        let b = self.is_prop_of_uncached(e);
        self.ctx.rp.prop_cache.insert(e, b);
        b
    }

    fn is_prop_of_uncached(&mut self, e: ExprPtr<'t>) -> bool {
        let sort = self.infer_clo(Clo::of(e), InferOnly);
        let v = self.nb_of_clo(Clo::of(sort));
        let v = self.nb_force(0, v);
        let v = self.nb_whnf(0, v);
        match self.ctx.nb.get(v) {
            crate::nbe::Value::Sort { level } => self.ctx.is_zero(level),
            _ => false,
        }
    }

    pub fn is_prop(&mut self, e: ExprPtr<'t>) -> (bool, ExprPtr<'t>) {
        let ty = self.infer_then_whnf(e, InferOnly);
        match self.ctx.read_expr(ty) {
            Sort { level, .. } => (self.ctx.is_zero(level), ty),
            _ => (false, ty),
        }
    }


    pub fn is_proof(&mut self, e: ExprPtr<'t>) -> (bool, ExprPtr<'t>) {
        let infd = self.infer(e, InferOnly);
        (self.is_prop(infd).0, infd)
    }

    // ---- whnf ----

    pub(crate) fn mk_sclo(&self, c: Clo<'t>) -> SClo<'t> {
        let mut spine = SpineVec::new();
        let mut e = c.e;
        while let App { fun, arg, .. } = self.ctx.read_expr(e) {
            spine.push(Clo { e: arg, env: c.env });
            e = fun;
        }
        spine.reverse();
        SClo { head: Clo { e, env: c.env }, spine }
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



    #[inline]

    

    


    /// The value denoted by a closure: evaluation reads the checker's own
    /// environment, so nothing is translated. Memoized on the closure: the
    /// same term recurs at many def-eq entries, and this is the one place a
    /// whole skeleton would otherwise be rewalked.
    pub(crate) fn nb_of_clo(&mut self, c: Clo<'t>) -> crate::nbe::ValId {
        if let Some(&v) = self.ctx.nb.clo_val_cache.get(&(c.e, c.env)) {
            return v;
        }
        let v = self.nb_eval(0, c.env, c.e);
        self.ctx.nb.clo_val_cache.insert((c.e, c.env), v);
        v
    }













    /// `Ok(b)` decided; `Err((tn, sn))` both irreducible.
    #[allow(clippy::type_complexity)]










    /// Could `ty` be a proposition? A universe parameter stands for a level
    /// that an instantiation may send to zero, so it counts.
    fn may_be_prop_of(&mut self, ty: ExprPtr<'t>) -> bool {
        let sort = self.infer_clo(Clo::of(ty), InferOnly);
        let v = self.nb_of_clo(Clo::of(sort));
        let v = self.nb_force(0, v);
        let v = self.nb_whnf(0, v);
        match self.ctx.nb.get(v) {
            crate::nbe::Value::Sort { level } => self.ctx.may_be_prop(level),
            _ => false,
        }
    }


    // ---- inference ----

    pub(crate) fn infer_clo(&mut self, c: Clo<'t>, flag: InferFlag) -> ExprPtr<'t> {
        self.infer_go(c, flag)
    }

    pub(crate) fn is_def_eq(&mut self, t: Clo<'t>, s: Clo<'t>) -> bool {
        self.ctx.rp.ctrs[3] += 1;
        // syntactically equal modulo substitution: answered without
        // evaluating either side
        if self.eq_mod(t.e, t.env, 0, s.e, s.env, 0) {
            return true;
        }
        let a = self.nb_of_clo(t);
        let b = self.nb_of_clo(s);
        self.nb_conv(0, a, b)
    }

    fn infer_go(&mut self, c: Clo<'t>, flag: InferFlag) -> ExprPtr<'t> {
        self.ctx.rp.ctrs[0] += 1;
        let n = self.ctx.read_expr(c.e);
        match n {
            Var { dbj_idx, .. } => match self.lookup(c.env, dbj_idx) {
                Entry::Neu(fv) => {
                    self.ctx.rp.ctrs[19] += 1;
                    self.fvar_type(fv)
                }
                Entry::Val(e2, env2) => {
                    self.ctx.rp.ctrs[18] += 1;
                    self.infer_clo(Clo { e: e2, env: env2 }, flag)
                }
                Entry::V(v) => {
                    self.ctx.rp.ctrs[18] += 1;
                    let ty = self.nb_type(0, v);
                    self.nb_readback(ty)
                }
            },
            Local { binder_type, .. } => {
                self.ctx.rp.ctrs[20] += 1;
                binder_type
            }
            Sort { level, .. } => {
                self.ctx.rp.ctrs[21] += 1;
                if flag == Check {
                    self.check_level(level);
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
                                self.check_level(l);
                            }
                        }
                        return r;
                    }
                    let r = self.infer_const(name, levels, flag);
                    self.ctx.rp.g_inst_ty.insert(c.e, r);
                    return r;
                }
                self.infer_const(name, levels, flag)
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
                let key = self.key(c);
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
                    Lambda { .. } => self.infer_lambda(c, flag),
                    Pi { .. } => self.infer_pi(c, flag),
                    Let { .. } => self.infer_let(c, flag),
                    App { .. } => {
                        let s = self.mk_sclo(c);
                        self.infer_s(&s, flag)
                    }
                    Proj { ty_name, idx, structure, .. } => {
                        self.infer_proj(ty_name, idx, Clo { e: structure, env: c.env }, flag)
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
    fn check_level(&mut self, l: LevelPtr<'t>) {
        if let Some(di) = self.declar_info {
            assert!(
                self.ctx.all_uparams_defined(l, di.uparams),
                "undefined universe parameter"
            );
        }
    }

    fn infer_const(
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
                self.check_level(l);
            }
        }
        self.ctx.subst_declar_info_levels(info, levels)
    }


    fn infer_lambda(&mut self, c: Clo<'t>, flag: InferFlag) -> ExprPtr<'t> {
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
            let d = self.reify(Clo { e: binder_type, env });
            if flag == Check {
                let dty = self.infer_clo(Clo::of(d), flag);
                self.ensure_sort_clo(dty);
            }
            let level = self.next_level(e, env).max(self.max_level(d));
            if binders.is_empty() {
                start = level;
            }
            let fv = self.fvar_at(level, d);
            env = self.push_entry(env, Entry::Neu(fv));
            binders.push((binder_name, binder_style, d));
            e = body;
        }
        let bt = self.infer_clo(Clo { e, env }, flag);
        let bt = self.cheap_beta_reduce(bt);
        let n = u32::try_from(binders.len()).unwrap();
        let mut r = self.ctx.abstr_levels_at(bt, start, start + n);
        for (i, (binder_name, binder_style, d)) in binders.into_iter().enumerate().rev() {
            let i = u32::try_from(i).unwrap();
            let d = self.ctx.abstr_levels_at(d, start, start + i);
            r = self.ctx.mk_pi(binder_name, binder_style, d, r);
        }
        r
    }

    fn infer_pi(&mut self, c: Clo<'t>, flag: InferFlag) -> ExprPtr<'t> {
        let Pi { binder_type, body, .. } = self.ctx.read_expr(c.e) else {
            unreachable!()
        };
        let d = self.reify(Clo { e: binder_type, env: c.env });
        let dty = self.infer_clo(Clo::of(d), flag);
        let u = self.ensure_sort_clo(dty);
        let level = self.next_level(c.e, c.env).max(self.max_level(d));
        let fv = self.fvar_at(level, d);
        let env2 = self.push_entry(c.env, Entry::Neu(fv));
        let bt = self.infer_clo(Clo { e: body, env: env2 }, flag);
        let s = self.ensure_sort_clo(bt);
        // mkLevelIMax': imax with immediate simplifications
        let lvl = self.ctx.imax(u, s);
        let lvl = self.ctx.simplify(lvl);
        self.ctx.mk_sort(lvl)
    }


    fn infer_let(&mut self, c: Clo<'t>, flag: InferFlag) -> ExprPtr<'t> {
        let Let { binder_type, val, body, .. } = self.ctx.read_expr(c.e) else {
            unreachable!()
        };
        let t = self.reify(Clo { e: binder_type, env: c.env });
        if flag == Check {
            let tty = self.infer_clo(Clo::of(t), flag);
            self.ensure_sort_clo(tty);
            let vty = self.infer_clo(Clo { e: val, env: c.env }, flag);
            assert!(
                self.is_def_eq(Clo::of(vty), Clo::of(t)),
                "let type mismatch"
            );
        }
        let env2 = self.push_entry(c.env, Entry::Val(val, c.env));
        self.infer_clo(Clo { e: body, env: env2 }, flag)
    }

    pub(crate) fn infer_s(&mut self, s: &SClo<'t>, flag: InferFlag) -> ExprPtr<'t> {
        if s.spine.is_empty() {
            return self.infer_clo(s.head, flag);
        }
        let mut f_ty: Clo<'t> = Clo::of(self.infer_clo(s.head, flag));
        for &arg in s.spine.iter() {
            // A syntactic binder is peeled without evaluating its domain;
            // anything else is forced to a Pi value. `dom_e` carries the
            // domain as a closure over `pi_env`, `dom_v` as a value.
            let (dom_e, dom_v, pi_env, body);
            if let Pi { binder_type, body: b, .. } = self.ctx.read_expr(f_ty.e) {
                dom_e = Some(binder_type);
                dom_v = None;
                pi_env = f_ty.env;
                body = b;
            } else {
                let fv = self.nb_of_clo(f_ty);
                let fv = self.nb_force(0, fv);
                let fv = self.nb_whnf(0, fv);
                let crate::nbe::Value::Pi { domain, env, body: b, .. } =
                    self.ctx.nb.get(fv)
                else {
                    panic!("function expected");
                };
                dom_e = None;
                dom_v = Some(domain);
                pi_env = env;
                body = b;
            }
            if flag == Check {
                let a_ty = self.infer_clo(arg, flag);
                // `@eagerReduce A a` in argument position asks for the
                // comparison to reduce without waiting for both sides to be
                // free of free variables.
                let outer_eager = self.ctx.eager_mode;
                if self.ctx.is_eager_reduce_app(arg.e) {
                    self.ctx.eager_mode = true;
                }
                let ok = if let (None, Some(bt)) = (dom_v, dom_e) {
                    self.is_def_eq(Clo { e: bt, env: pi_env }, Clo::of(a_ty))
                } else {
                    let dom_v = dom_v.expect("domain");
                    let a_ty_v = self.nb_of_clo(Clo::of(a_ty));
                    self.nb_conv(0, a_ty_v, dom_v)
                };
                self.ctx.eager_mode = outer_eager;
                assert!(ok, "application type mismatch");
            }
            let env2 = self.push_entry(pi_env, Entry::Val(arg.e, arg.env));
            f_ty = Clo { e: body, env: env2 };
        }
        let r = self.reify(f_ty);
        r
    }

    fn infer_proj(
        &mut self,
        type_name: NamePtr<'t>,
        idx: usize,
        strukt: Clo<'t>,
        flag: InferFlag,
    ) -> ExprPtr<'t> {
        let s_ty = self.infer_clo(strukt, flag);
        let st_v = self.nb_of_clo(Clo::of(s_ty));
        let st_v = self.nb_force(0, st_v);
        let st_v = self.nb_whnf(0, st_v);
        let crate::nbe::Value::Rigid { head, spine: st_spine } = self.ctx.nb.get(st_v) else {
            panic!("invalid projection");
        };
        let crate::nbe::RigidHead::Const(_, i_name, i_levels) = head else {
            panic!("invalid projection");
        };
        assert!(i_name == type_name, "invalid projection");
        let (num_params, num_indices, ctors) = match self.env.get_inductive(&i_name) {
            Some(ind) => (ind.num_params, ind.num_indices, ind.all_ctor_names.clone()),
            None => panic!("invalid projection"),
        };
        assert!(ctors.len() == 1, "invalid projection");
        let st_args: Vec<crate::nbe::ValId> = self
            .ctx
            .nb
            .spine_to_vec(st_spine)
            .into_iter()
            .map(|el| match el {
                crate::nbe::Elim::App(a) => a,
                _ => panic!("invalid projection"),
            })
            .collect();
        assert!(
            st_args.len() == (num_params + num_indices) as usize,
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
            let rv = self.nb_of_clo(r);
            let rv = self.nb_force(0, rv);
            let rv = self.nb_whnf(0, rv);
            let crate::nbe::Value::Pi { env: pi_env, body, .. } = self.ctx.nb.get(rv) else {
                panic!("invalid projection");
            };
            let env2 = self.push_entry_v(pi_env, st_args[pi]);
            r = Clo { e: body, env: env2 };
        }
        let is_prop_ty = self.may_be_prop_of(s_ty);
        for fi in 0..idx {
            let rv = self.nb_of_clo(r);
            let rv = self.nb_force(0, rv);
            let rv = self.nb_whnf(0, rv);
            let crate::nbe::Value::Pi { domain, env: pi_env, body, .. } =
                self.ctx.nb.get(rv)
            else {
                panic!("invalid projection");
            };
            if self.lbr(body) > 0 && is_prop_ty {
                let d = self.nb_readback(domain);
                assert!(self.is_prop_of(d), "infer_proj prop");
            }
            let bv = self.ctx.mk_var(0);
            let proj = self.ctx.mk_proj(i_name, fi, bv);
            let senv = self.push_entry(ENV_NIL, Entry::Val(strukt.e, strukt.env));
            let env2 = self.push_entry(pi_env, Entry::Val(proj, senv));
            r = Clo { e: body, env: env2 };
        }
        let rv = self.nb_of_clo(r);
        let rv = self.nb_force(0, rv);
        let rv = self.nb_whnf(0, rv);
        let crate::nbe::Value::Pi { domain, .. } = self.ctx.nb.get(rv) else {
            panic!("invalid projection");
        };
        let d = self.nb_readback(domain);
        if is_prop_ty {
            assert!(self.is_prop_of(d), "infer_proj prop");
        }
        d
    }

    fn cheap_beta_reduce(&mut self, e: ExprPtr<'t>) -> ExprPtr<'t> {
        if !matches!(self.ctx.read_expr(e), App { .. }) {
            return e;
        }
        let s = self.mk_sclo(Clo::of(e));
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

}
