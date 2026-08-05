#![allow(dead_code)] // wired in as the evaluator is completed
//! Evaluation of expressions to values.
//!
//! `eval` reads an expression under an environment and produces a value. It
//! never rebuilds an expression: entering a binder records the argument in
//! the environment, and an argument is recorded unevaluated so that one that
//! is never looked at is never evaluated. Applying a value that is stuck
//! extends its spine rather than reconstructing the application.

use crate::env::Declar;
use crate::expr::Expr;
use crate::nbe::{Elim, RigidHead, SpineId, ValId, Value, VEnvId, SPINE_EMPTY, VENV_NIL};
use crate::tc::TypeChecker;
use crate::util::ExprPtr;
use Expr::*;

impl<'x, 't: 'x, 'p: 't> TypeChecker<'x, 't, 'p> {
    /// The value of `e` under `env`.
    pub(crate) fn nb_eval(&mut self, env: VEnvId, e: ExprPtr<'t>) -> ValId {
        // A closed expression means the same thing under every environment,
        // so it is evaluated once and shared.
        let closed = self.ctx.num_loose_bvars(e) == 0;
        let env = if closed { VENV_NIL } else { env };
        if let Some(&v) = self.ctx.nb.eval_cache.get(&(e, env)) {
            return v;
        }
        let v = self.nb_eval_go(env, e);
        self.ctx.nb.eval_cache.insert((e, env), v);
        v
    }

    fn nb_eval_go(&mut self, env: VEnvId, e: ExprPtr<'t>) -> ValId {
        match self.ctx.read_expr(e) {
            Var { dbj_idx, .. } => match self.ctx.nb.venv_lookup(env, u32::from(dbj_idx)) {
                Some(v) => v,
                None => panic!("eval: loose bound variable"),
            },
            Sort { level, .. } => {
                let level = self.ctx.simplify(level);
                self.ctx.nb.alloc(Value::Sort { level })
            }
            NatLit { ptr, .. } => self.ctx.nb.alloc(Value::NatLit { ptr }),
            StringLit { ptr, .. } => self.ctx.nb.alloc(Value::StrLit { ptr }),
            Local { .. } => self.ctx.nb.mk_rigid(RigidHead::Local(e), SPINE_EMPTY),
            Const { name, levels, .. } => self.nb_const(name, levels),
            App { fun, arg, .. } => {
                let f = self.nb_eval(env, fun);
                let a = self.nb_thunk(env, arg);
                self.nb_apply(f, a)
            }
            Lambda { binder_name, binder_style, binder_type, body, .. } => {
                self.ctx.nb.alloc(Value::Lam {
                    binder_name,
                    binder_style,
                    binder_type,
                    domain: None,
                    env,
                    body,
                })
            }
            Pi { binder_name, binder_style, binder_type, body, .. } => {
                let domain = self.nb_eval(env, binder_type);
                self.ctx.nb.alloc(Value::Pi { binder_name, binder_style, domain, env, body })
            }
            Let { val, body, .. } => {
                let v = self.nb_thunk(env, val);
                let env2 = self.ctx.nb.venv_cons(env, v);
                self.nb_eval(env2, body)
            }
            Proj { ty_name, idx, structure, .. } => {
                let s = self.nb_eval(env, structure);
                self.nb_proj(ty_name, idx, s)
            }
        }
    }

    /// Record an argument without evaluating it. Something already in normal
    /// form is evaluated straight away, since a thunk over it would cost more
    /// than the value.
    fn nb_thunk(&mut self, env: VEnvId, e: ExprPtr<'t>) -> ValId {
        match self.ctx.read_expr(e) {
            Var { .. } | Sort { .. } | NatLit { .. } | StringLit { .. } | Local { .. } => {
                self.nb_eval(env, e)
            }
            _ => {
                let env = if self.ctx.num_loose_bvars(e) == 0 { VENV_NIL } else { env };
                self.ctx.nb.alloc(Value::Thunk { env, expr: e, forced: None })
            }
        }
    }

    /// A constant: one that has a body can still unfold, anything else is
    /// stuck on its name.
    fn nb_const(&mut self, name: crate::util::NamePtr<'t>, levels: crate::util::LevelsPtr<'t>) -> ValId {
        let unfoldable = matches!(
            self.env.get_declar(&name),
            Some(Declar::Definition { .. } | Declar::Theorem { .. })
        );
        if unfoldable {
            self.ctx.nb.alloc(Value::Unfold { name, levels, spine: SPINE_EMPTY, forced: None })
        } else {
            self.ctx.nb.mk_rigid(RigidHead::Const(name, levels), SPINE_EMPTY)
        }
    }

    /// Force a thunk, remembering the result in it so it is evaluated once.
    pub(crate) fn nb_force(&mut self, v: ValId) -> ValId {
        match self.ctx.nb.get(v) {
            Value::Thunk { env, expr, forced } => {
                if let Some(f) = forced {
                    return f;
                }
                let f = self.nb_eval(env, expr);
                let f = self.nb_force(f);
                if let Value::Thunk { forced, .. } = &mut self.ctx.nb.vals[v as usize] {
                    *forced = Some(f);
                }
                f
            }
            _ => v,
        }
    }

    /// Apply a value to an argument.
    pub(crate) fn nb_apply(&mut self, f: ValId, a: ValId) -> ValId {
        let f = self.nb_force(f);
        match self.ctx.nb.get(f) {
            Value::Lam { env, body, .. } => {
                let env2 = self.ctx.nb.venv_cons(env, a);
                self.nb_eval(env2, body)
            }
            Value::Rigid { head, spine } => {
                let spine = self.ctx.nb.spine_snoc(spine, Elim::App(a));
                self.ctx.nb.mk_rigid(head, spine)
            }
            Value::Unfold { name, levels, spine, .. } => {
                let spine = self.ctx.nb.spine_snoc(spine, Elim::App(a));
                self.ctx.nb.alloc(Value::Unfold { name, levels, spine, forced: None })
            }
            _ => panic!("apply: not a function"),
        }
    }

    /// Project a field out of a value.
    fn nb_proj(&mut self, ty_name: crate::util::NamePtr<'t>, idx: usize, s: ValId) -> ValId {
        let s = self.nb_force(s);
        match self.ctx.nb.get(s) {
            Value::Rigid { head, spine } => {
                let spine = self.ctx.nb.spine_snoc(spine, Elim::Proj { ty_name, idx });
                self.ctx.nb.mk_rigid(head, spine)
            }
            Value::Unfold { name, levels, spine, .. } => {
                let spine = self.ctx.nb.spine_snoc(spine, Elim::Proj { ty_name, idx });
                self.ctx.nb.alloc(Value::Unfold { name, levels, spine, forced: None })
            }
            _ => {
                let spine = self.ctx.nb.spine_snoc(SPINE_EMPTY, Elim::Proj { ty_name, idx });
                let _ = spine;
                panic!("proj: not a structure")
            }
        }
    }

    /// The body of a constant, with its level parameters replaced, evaluated
    /// once per constant and shared by every occurrence.
    pub(crate) fn nb_unfold_head(
        &mut self,
        name: crate::util::NamePtr<'t>,
        levels: crate::util::LevelsPtr<'t>,
    ) -> Option<ValId> {
        if let Some(&v) = self.ctx.nb.unfold_cache.get(&(name, levels)) {
            return v;
        }
        let r = self.env.get_declar_val(&name).map(|(uparams, val)| {
            let val = self.ctx.subst_expr_levels(val, uparams, levels);
            self.nb_eval(VENV_NIL, val)
        });
        self.ctx.nb.unfold_cache.insert((name, levels), r);
        r
    }

    /// Apply a spine to a value.
    pub(crate) fn nb_apply_spine(&mut self, mut v: ValId, spine: SpineId) -> ValId {
        for elim in self.ctx.nb.spine_to_vec(spine) {
            v = match elim {
                Elim::App(a) => self.nb_apply(v, a),
                Elim::Proj { ty_name, idx } => self.nb_proj(ty_name, idx, v),
            };
        }
        v
    }
}
