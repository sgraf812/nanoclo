#![allow(dead_code)] // wired in as the evaluator is completed
//! Values, environments and spines for evaluation.
//!
//! A value is the result of evaluating an expression under an environment,
//! kept as a graph rather than rebuilt as an expression. Values, their
//! environments and their spines live in arenas and are named by index, so
//! two of them are the same when their indices are equal, and asking whether
//! two values are the same term starts as an integer comparison.
//!
//! An environment maps a de Bruijn index to a value, so it is the delayed
//! substitution the checker was already carrying, with the entries evaluated
//! rather than described. An argument enters as a thunk and is evaluated at
//! most once, and a definition's body is evaluated at most once per
//! constant, so work done under one occurrence is not redone under another.

use crate::expr::BinderStyle;
use crate::util::{BigUintPtr, ExprPtr, LevelPtr, LevelsPtr, NamePtr, StringPtr};

pub(crate) type ValId = u32;
pub(crate) type VEnvId = u32;
pub(crate) type SpineId = u32;

/// The empty environment.
pub(crate) const VENV_NIL: VEnvId = 0;
/// The spine of a value applied to nothing.
pub(crate) const SPINE_EMPTY: SpineId = 0;

/// What a neutral value is stuck on.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RigidHead<'t> {
    /// A variable standing for an opened binder, named by its de Bruijn level
    /// and carrying the value of its type.
    BVar(u32, ValId),
    /// A free variable from the surrounding declaration.
    Local(ExprPtr<'t>),
    /// A constant that cannot unfold: an axiom, constructor, recursor,
    /// quotient operation or inductive type.
    Const(NamePtr<'t>, LevelsPtr<'t>),
}

/// One step of elimination applied to a neutral value.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Elim<'t> {
    App(ValId),
    Proj { ty_name: NamePtr<'t>, idx: usize },
}

#[derive(Clone, Copy)]
pub(crate) enum Value<'t> {
    /// Stuck: a head that cannot reduce, under a spine of eliminations.
    Rigid { head: RigidHead<'t>, spine: SpineId },
    /// A constant that can unfold, kept unfolded until someone needs it.
    /// `forced` holds the result once it has been asked for.
    Unfold {
        name: NamePtr<'t>,
        levels: LevelsPtr<'t>,
        spine: SpineId,
        forced: Option<ValId>,
    },
    /// `binder_type` is the unevaluated domain; `domain` holds its value once
    /// a comparison has needed it.
    Lam {
        binder_name: NamePtr<'t>,
        binder_style: BinderStyle,
        binder_type: ExprPtr<'t>,
        domain: Option<ValId>,
        env: VEnvId,
        body: ExprPtr<'t>,
    },
    Pi {
        binder_name: NamePtr<'t>,
        binder_style: BinderStyle,
        domain: ValId,
        env: VEnvId,
        body: ExprPtr<'t>,
    },
    Sort { level: LevelPtr<'t> },
    NatLit { ptr: BigUintPtr<'t> },
    StrLit { ptr: StringPtr<'t> },
    /// An argument that has not been needed yet.
    Thunk { env: VEnvId, expr: ExprPtr<'t>, forced: Option<ValId> },
}

pub(crate) struct VEnvNode {
    pub(crate) v: ValId,
    pub(crate) parent: VEnvId,
    pub(crate) len: u32,
}

pub(crate) struct SpineNode<'t> {
    pub(crate) elim: Elim<'t>,
    pub(crate) parent: SpineId,
    pub(crate) len: u32,
}

/// The arenas, and the tables that give equal constructions equal indices.
pub(crate) struct Vals<'t> {
    pub(crate) vals: Vec<Value<'t>>,
    pub(crate) venvs: Vec<VEnvNode>,
    pub(crate) spines: Vec<SpineNode<'t>>,
    /// `(parent, value) -> environment`
    pub(crate) venv_intern: crate::util::FxHashMap<(VEnvId, ValId), VEnvId>,
    /// `(parent, elimination) -> spine`
    pub(crate) spine_intern: crate::util::FxHashMap<(SpineId, u64, u64), SpineId>,
    /// `(level, type) -> the variable standing for a binder at that level`
    pub(crate) bvar_intern: crate::util::FxHashMap<(u32, ValId), ValId>,
    /// `(head, spine) -> the neutral value`
    pub(crate) rigid_intern: crate::util::FxHashMap<(u64, u64, SpineId), ValId>,
    /// `(expression, environment) -> its value`
    pub(crate) eval_cache: crate::util::FxHashMap<(ExprPtr<'t>, VEnvId), ValId>,
    /// `(constant, levels) -> the value of its body`
    pub(crate) unfold_cache: crate::util::FxHashMap<(NamePtr<'t>, LevelsPtr<'t>), Option<ValId>>,
}

impl<'t> Vals<'t> {
    pub(crate) fn new() -> Self {
        Vals {
            // index 0 of each arena is the empty case, so that VENV_NIL and
            // SPINE_EMPTY are valid indices and need no special casing.
            vals: Vec::new(),
            venvs: vec![VEnvNode { v: 0, parent: 0, len: 0 }],
            spines: vec![SpineNode {
                elim: Elim::Proj { ty_name: crate::util::Ptr::from(crate::util::DagMarker::ExportFile, 0), idx: 0 },
                parent: 0,
                len: 0,
            }],
            venv_intern: crate::util::new_fx_hash_map(),
            spine_intern: crate::util::new_fx_hash_map(),
            bvar_intern: crate::util::new_fx_hash_map(),
            rigid_intern: crate::util::new_fx_hash_map(),
            eval_cache: crate::util::new_fx_hash_map(),
            unfold_cache: crate::util::new_fx_hash_map(),
        }
    }

    pub(crate) fn reset_decl(&mut self) {
        self.vals.clear();
        self.venvs.truncate(1);
        self.spines.truncate(1);
        self.venv_intern.clear();
        self.spine_intern.clear();
        self.bvar_intern.clear();
        self.rigid_intern.clear();
        self.eval_cache.clear();
        self.unfold_cache.clear();
    }

    #[inline]
    pub(crate) fn get(&self, v: ValId) -> Value<'t> { self.vals[v as usize] }

    #[inline]
    pub(crate) fn alloc(&mut self, v: Value<'t>) -> ValId {
        let id = u32::try_from(self.vals.len()).expect("value arena overflow");
        self.vals.push(v);
        id
    }

    #[inline]
    pub(crate) fn venv_len(&self, e: VEnvId) -> u32 { self.venvs[e as usize].len }

    #[inline]
    pub(crate) fn spine_len(&self, s: SpineId) -> u32 { self.spines[s as usize].len }

    /// Extend an environment. Interned, so an environment built twice the
    /// same way is the same environment.
    pub(crate) fn venv_cons(&mut self, parent: VEnvId, v: ValId) -> VEnvId {
        if let Some(&e) = self.venv_intern.get(&(parent, v)) {
            return e;
        }
        let len = self.venvs[parent as usize].len + 1;
        let id = u32::try_from(self.venvs.len()).expect("environment arena overflow");
        self.venvs.push(VEnvNode { v, parent, len });
        self.venv_intern.insert((parent, v), id);
        id
    }

    /// The value bound to a de Bruijn index, counting from the innermost.
    pub(crate) fn venv_lookup(&self, mut e: VEnvId, mut idx: u32) -> Option<ValId> {
        loop {
            let node = &self.venvs[e as usize];
            if node.len == 0 {
                return None;
            }
            if idx == 0 {
                return Some(node.v);
            }
            idx -= 1;
            e = node.parent;
        }
    }

    pub(crate) fn spine_snoc(&mut self, parent: SpineId, elim: Elim<'t>) -> SpineId {
        let key = match elim {
            Elim::App(v) => (parent, u64::from(v) << 1, 0u64),
            Elim::Proj { ty_name, idx } => {
                (parent, ty_name.get_hash() << 1 | 1, idx as u64 + 1)
            }
        };
        if let Some(&s) = self.spine_intern.get(&key) {
            return s;
        }
        let len = self.spines[parent as usize].len + 1;
        let id = u32::try_from(self.spines.len()).expect("spine arena overflow");
        self.spines.push(SpineNode { elim, parent, len });
        self.spine_intern.insert(key, id);
        id
    }

    /// The eliminations of a spine, outermost last.
    pub(crate) fn spine_to_vec(&self, mut s: SpineId) -> Vec<Elim<'t>> {
        let mut out = Vec::with_capacity(self.spine_len(s) as usize);
        while self.spines[s as usize].len != 0 {
            let node = &self.spines[s as usize];
            out.push(node.elim);
            s = node.parent;
        }
        out.reverse();
        out
    }

    /// The variable standing for a binder opened at `level`. Interned on the
    /// level and the type, so two openings of the same binder are the same
    /// variable and everything built over them coincides.
    pub(crate) fn mk_bvar(&mut self, level: u32, ty: ValId) -> ValId {
        if let Some(&v) = self.bvar_intern.get(&(level, ty)) {
            return v;
        }
        let v = self.alloc(Value::Rigid {
            head: RigidHead::BVar(level, ty),
            spine: SPINE_EMPTY,
        });
        self.bvar_intern.insert((level, ty), v);
        v
    }

    /// A neutral value, interned so that the same head under the same spine
    /// is one value.
    pub(crate) fn mk_rigid(&mut self, head: RigidHead<'t>, spine: SpineId) -> ValId {
        let key = match head {
            RigidHead::BVar(l, ty) => (u64::from(l) << 2, u64::from(ty), spine),
            RigidHead::Local(e) => (e.get_hash() << 2 | 1, 0, spine),
            RigidHead::Const(n, ls) => (n.get_hash() << 2 | 2, ls.get_hash(), spine),
        };
        if let Some(&v) = self.rigid_intern.get(&key) {
            return v;
        }
        let v = self.alloc(Value::Rigid { head, spine });
        self.rigid_intern.insert(key, v);
        v
    }
}
