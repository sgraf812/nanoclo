use crate::env::{Declar, DeclarInfo, Env};
use crate::expr::Expr;
use crate::level::Level;
use crate::util::{
    ExportFile, ExprPtr, LevelPtr, NamePtr, TcCtx
};
use std::error::Error;

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
                    for (i, declar) in self.declars.values().enumerate() {
                        if i < skip {
                            continue
                        }
                        if let Some(stop) = stop_after {
                            if i > stop {
                                break
                            }
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











    pub fn is_sort_zero(&mut self, e: ExprPtr<'t>) -> bool {
        let e = self.whnf(e);
        match self.ctx.read_expr(e) {
            Sort { level, .. } => self.ctx.read_level(level) == Level::Zero,
            _ => false,
        }
    }
    pub fn is_proposition(&mut self, e: ExprPtr<'t>) -> (bool, ExprPtr<'t>) {
        let infd = self.infer(e, InferOnly);
        (self.is_sort_zero(infd), infd)
    }

    pub fn is_proof(&mut self, e: ExprPtr<'t>) -> (bool, ExprPtr<'t>) {
        let infd = self.infer(e, InferOnly);
        (self.is_proposition(infd).0, infd)
    }



}
