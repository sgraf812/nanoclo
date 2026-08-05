//! Values, environments and spines for evaluation.
//!
//! A value is the result of evaluating an expression under an environment,
//! kept as a graph rather than rebuilt as an expression. Values, their
//! environments and their spines live in arenas and are named by index, and
//! every construction goes through an intern table, so two of them are the
//! same when their indices are equal and asking whether two values are the
//! same term is an integer comparison.
//!
//! An environment maps a de Bruijn index to a value, so it is the delayed
//! substitution the checker was already carrying, with the entries evaluated
//! rather than described. An argument enters as a thunk and is evaluated at
//! most once, and a definition's body is evaluated at most once per
//! constant, so work done under one occurrence is not redone under another.

use crate::expr::BinderStyle;
use crate::util::{
    new_fx_hash_map, new_fx_hash_set, BigUintPtr, ExprPtr, FxHashMap, FxHashSet, LevelPtr,
    LevelsPtr, NamePtr, StringPtr,
};

pub(crate) type ValId = u32;
pub(crate) type VEnvId = u32;
pub(crate) type SpineId = u32;

/// The empty environment.
pub(crate) const VENV_NIL: VEnvId = 0;
/// The spine of a value applied to nothing.
pub(crate) const SPINE_EMPTY: SpineId = 0;

/// How a constant behaves under reduction.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum ConstKind {
    /// An axiom or opaque definition: no body to unfold to.
    Axiom,
    Ctor,
    /// A recursor, which reduces once its major premise is a constructor.
    Recursor,
    /// `Quot.lift` or `Quot.ind`, which reduce once their argument is
    /// `Quot.mk`; also `Quot` and `Quot.mk` themselves, which never reduce.
    QuotConst,
    Inductive,
}

/// What a neutral value is stuck on.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RigidHead<'t> {
    /// A variable standing for an opened binder, named by its de Bruijn level
    /// and carrying the value of its type.
    BVar(u32, ValId),
    /// A free variable from the surrounding declaration.
    Local(ExprPtr<'t>),
    Const(ConstKind, NamePtr<'t>, LevelsPtr<'t>),
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
    /// A constant with a body, kept folded until someone needs it. `forced`
    /// holds the result of unfolding it once that has been asked for.
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

/// The arenas, the tables that give equal constructions equal indices, and
/// the memos over them.
pub(crate) struct Vals<'t> {
    pub(crate) vals: Vec<Value<'t>>,
    pub(crate) venvs: Vec<VEnvNode>,
    pub(crate) spines: Vec<SpineNode<'t>>,

    // ---- interning ----
    /// `(parent, value) -> environment`
    venv_intern: FxHashMap<(VEnvId, ValId), VEnvId>,
    /// `(parent, elimination) -> spine`
    spine_intern: FxHashMap<(SpineId, Elim<'t>), SpineId>,
    /// `(head, spine) -> the neutral value`
    rigid_intern: FxHashMap<(RigidHead<'t>, SpineId), ValId>,
    /// `(constant, levels, spine) -> the folded application`
    unfold_intern: FxHashMap<(NamePtr<'t>, LevelsPtr<'t>, SpineId), ValId>,
    /// `(domain expression, environment, body) -> the lambda`
    lam_intern: FxHashMap<(ExprPtr<'t>, VEnvId, ExprPtr<'t>), ValId>,
    /// `(domain value, environment, body) -> the pi`
    pi_intern: FxHashMap<(ValId, VEnvId, ExprPtr<'t>), ValId>,
    sort_intern: FxHashMap<LevelPtr<'t>, ValId>,
    nat_intern: FxHashMap<BigUintPtr<'t>, ValId>,
    str_intern: FxHashMap<StringPtr<'t>, ValId>,
    /// `(environment, expression) -> the thunk over it`
    thunk_intern: FxHashMap<(VEnvId, ExprPtr<'t>), ValId>,

    // ---- memos ----
    /// `(expression, environment) -> its value`
    pub(crate) eval_cache: FxHashMap<(ExprPtr<'t>, VEnvId), ValId>,
    /// `(constant, levels) -> the value of its body`
    pub(crate) unfold_cache: FxHashMap<(NamePtr<'t>, LevelsPtr<'t>), Option<ValId>>,
    /// `(constant, levels) -> the value denoting it`
    pub(crate) const_val_cache: FxHashMap<(NamePtr<'t>, LevelsPtr<'t>), ValId>,
    /// `(constant, levels) -> the value of its type`
    pub(crate) const_ty_cache: FxHashMap<(NamePtr<'t>, LevelsPtr<'t>), ValId>,
    /// `(constant, levels) -> the sort its type ends in`
    pub(crate) const_lvl_cache: FxHashMap<(NamePtr<'t>, LevelsPtr<'t>), Option<LevelPtr<'t>>>,
    /// `(recursor rule body, levels) -> the value of that body`
    pub(crate) rec_rule_cache: FxHashMap<(ExprPtr<'t>, LevelsPtr<'t>), ValId>,
    /// `stuck application -> what it reduces to`, `None` when it is stuck
    pub(crate) iota_cache: FxHashMap<ValId, Option<ValId>>,
    /// `(value, inductive) -> the constructor application it expands to`
    pub(crate) struct_eta_cache: FxHashMap<(ValId, NamePtr<'t>), Option<ValId>>,
    /// `value -> whether an opened binder or free variable occurs in it`
    pub(crate) open_cache: FxHashMap<ValId, bool>,
    /// `value -> the value of its type`
    pub(crate) type_cache: FxHashMap<ValId, ValId>,
    /// `local expression -> the value denoting it`
    pub(crate) local_cache: FxHashMap<ExprPtr<'t>, ValId>,
    /// `rapier environment -> the same bindings as values`
    pub(crate) clo_env_cache: FxHashMap<u32, VEnvId>,

    // ---- conversion results ----
    pub(crate) conv_pos: FxHashSet<(ValId, ValId)>,
    pub(crate) conv_neg: FxHashSet<(ValId, ValId)>,
    /// Failures recorded while comparing the arguments of two applications of
    /// the same constant. That comparison is a guess that may fail while the
    /// terms are still equal, so its failures are dropped when the guess is.
    pub(crate) conv_neg_probe: FxHashSet<(ValId, ValId)>,
    pub(crate) probe_depth: u32,
}

impl<'t> Vals<'t> {
    pub(crate) fn new() -> Self {
        Vals {
            vals: Vec::new(),
            // index 0 of each of these arenas is the empty case, so that
            // VENV_NIL and SPINE_EMPTY are valid indices needing no special
            // casing on the lookup paths.
            venvs: vec![VEnvNode { v: 0, parent: 0, len: 0 }],
            spines: vec![SpineNode {
                elim: Elim::Proj {
                    ty_name: crate::util::Ptr::from(crate::util::DagMarker::ExportFile, 0),
                    idx: 0,
                },
                parent: 0,
                len: 0,
            }],
            venv_intern: new_fx_hash_map(),
            spine_intern: new_fx_hash_map(),
            rigid_intern: new_fx_hash_map(),
            unfold_intern: new_fx_hash_map(),
            lam_intern: new_fx_hash_map(),
            pi_intern: new_fx_hash_map(),
            sort_intern: new_fx_hash_map(),
            nat_intern: new_fx_hash_map(),
            str_intern: new_fx_hash_map(),
            thunk_intern: new_fx_hash_map(),
            eval_cache: new_fx_hash_map(),
            unfold_cache: new_fx_hash_map(),
            const_val_cache: new_fx_hash_map(),
            const_ty_cache: new_fx_hash_map(),
            const_lvl_cache: new_fx_hash_map(),
            rec_rule_cache: new_fx_hash_map(),
            iota_cache: new_fx_hash_map(),
            struct_eta_cache: new_fx_hash_map(),
            open_cache: new_fx_hash_map(),
            type_cache: new_fx_hash_map(),
            local_cache: new_fx_hash_map(),
            clo_env_cache: new_fx_hash_map(),
            conv_pos: new_fx_hash_set(),
            conv_neg: new_fx_hash_set(),
            conv_neg_probe: new_fx_hash_set(),
            probe_depth: 0,
        }
    }

    /// Drop everything: values name arena positions, so nothing survives a
    /// declaration boundary. Capacity is kept where it is not extravagant.
    pub(crate) fn reset_decl(&mut self) {
        const CAP: usize = 1 << 16;
        fn rm<K: std::hash::Hash + Eq, V>(m: &mut FxHashMap<K, V>) {
            if m.capacity() > CAP {
                *m = new_fx_hash_map();
            } else if !m.is_empty() {
                m.clear();
            }
        }
        fn rs<K: std::hash::Hash + Eq>(m: &mut FxHashSet<K>) {
            if m.capacity() > CAP {
                *m = new_fx_hash_set();
            } else if !m.is_empty() {
                m.clear();
            }
        }
        if self.vals.capacity() > (1 << 20) {
            self.vals = Vec::new();
            self.venvs = vec![VEnvNode { v: 0, parent: 0, len: 0 }];
            let sentinel = self.spines.remove(0);
            self.spines = vec![sentinel];
        } else {
            self.vals.clear();
            self.venvs.truncate(1);
            self.spines.truncate(1);
        }
        rm(&mut self.venv_intern);
        rm(&mut self.spine_intern);
        rm(&mut self.rigid_intern);
        rm(&mut self.unfold_intern);
        rm(&mut self.lam_intern);
        rm(&mut self.pi_intern);
        rm(&mut self.sort_intern);
        rm(&mut self.nat_intern);
        rm(&mut self.str_intern);
        rm(&mut self.thunk_intern);
        rm(&mut self.eval_cache);
        rm(&mut self.unfold_cache);
        rm(&mut self.const_val_cache);
        rm(&mut self.const_ty_cache);
        rm(&mut self.const_lvl_cache);
        rm(&mut self.rec_rule_cache);
        rm(&mut self.iota_cache);
        rm(&mut self.struct_eta_cache);
        rm(&mut self.open_cache);
        rm(&mut self.type_cache);
        rm(&mut self.local_cache);
        rm(&mut self.clo_env_cache);
        rs(&mut self.conv_pos);
        rs(&mut self.conv_neg);
        rs(&mut self.conv_neg_probe);
        self.probe_depth = 0;
    }

    #[inline]
    pub(crate) fn get(&self, v: ValId) -> Value<'t> { self.vals[v as usize] }

    #[inline]
    fn alloc(&mut self, v: Value<'t>) -> ValId {
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
        if let Some(&s) = self.spine_intern.get(&(parent, elim)) {
            return s;
        }
        let len = self.spines[parent as usize].len + 1;
        let id = u32::try_from(self.spines.len()).expect("spine arena overflow");
        self.spines.push(SpineNode { elim, parent, len });
        self.spine_intern.insert((parent, elim), id);
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

    /// The arguments of a spine of applications, or `None` if a projection
    /// stands anywhere in it.
    pub(crate) fn spine_args(&self, mut s: SpineId) -> Option<Vec<ValId>> {
        let mut out = Vec::with_capacity(self.spine_len(s) as usize);
        while self.spines[s as usize].len != 0 {
            let node = &self.spines[s as usize];
            match node.elim {
                Elim::App(a) => out.push(a),
                Elim::Proj { .. } => return None,
            }
            s = node.parent;
        }
        out.reverse();
        Some(out)
    }

    /// The `i`th elimination counting from the innermost application.
    pub(crate) fn spine_get(&self, s: SpineId, i: usize) -> Option<Elim<'t>> {
        let len = self.spine_len(s) as usize;
        let mut steps = len.checked_sub(i + 1)?;
        let mut cur = s;
        loop {
            let node = &self.spines[cur as usize];
            if node.len == 0 {
                return None;
            }
            if steps == 0 {
                return Some(node.elim);
            }
            steps -= 1;
            cur = node.parent;
        }
    }

    // ---- interned constructors ----

    pub(crate) fn mk_rigid(&mut self, head: RigidHead<'t>, spine: SpineId) -> ValId {
        if let Some(&v) = self.rigid_intern.get(&(head, spine)) {
            return v;
        }
        let v = self.alloc(Value::Rigid { head, spine });
        self.rigid_intern.insert((head, spine), v);
        v
    }

    /// The variable standing for a binder opened at `level`. Interned on the
    /// level and the type, so two openings of the same binder are the same
    /// variable and everything built over them coincides.
    pub(crate) fn mk_bvar(&mut self, level: u32, ty: ValId) -> ValId {
        self.mk_rigid(RigidHead::BVar(level, ty), SPINE_EMPTY)
    }

    pub(crate) fn mk_unfold(
        &mut self,
        name: NamePtr<'t>,
        levels: LevelsPtr<'t>,
        spine: SpineId,
    ) -> ValId {
        if let Some(&v) = self.unfold_intern.get(&(name, levels, spine)) {
            return v;
        }
        let v = self.alloc(Value::Unfold { name, levels, spine, forced: None });
        self.unfold_intern.insert((name, levels, spine), v);
        v
    }

    pub(crate) fn mk_lam(
        &mut self,
        binder_name: NamePtr<'t>,
        binder_style: BinderStyle,
        binder_type: ExprPtr<'t>,
        env: VEnvId,
        body: ExprPtr<'t>,
    ) -> ValId {
        if let Some(&v) = self.lam_intern.get(&(binder_type, env, body)) {
            return v;
        }
        let v = self.alloc(Value::Lam {
            binder_name,
            binder_style,
            binder_type,
            domain: None,
            env,
            body,
        });
        self.lam_intern.insert((binder_type, env, body), v);
        v
    }

    pub(crate) fn mk_pi(
        &mut self,
        binder_name: NamePtr<'t>,
        binder_style: BinderStyle,
        domain: ValId,
        env: VEnvId,
        body: ExprPtr<'t>,
    ) -> ValId {
        if let Some(&v) = self.pi_intern.get(&(domain, env, body)) {
            return v;
        }
        let v = self.alloc(Value::Pi { binder_name, binder_style, domain, env, body });
        self.pi_intern.insert((domain, env, body), v);
        v
    }

    pub(crate) fn mk_sort(&mut self, level: LevelPtr<'t>) -> ValId {
        if let Some(&v) = self.sort_intern.get(&level) {
            return v;
        }
        let v = self.alloc(Value::Sort { level });
        self.sort_intern.insert(level, v);
        v
    }

    pub(crate) fn mk_nat(&mut self, ptr: BigUintPtr<'t>) -> ValId {
        if let Some(&v) = self.nat_intern.get(&ptr) {
            return v;
        }
        let v = self.alloc(Value::NatLit { ptr });
        self.nat_intern.insert(ptr, v);
        v
    }

    pub(crate) fn mk_str(&mut self, ptr: StringPtr<'t>) -> ValId {
        if let Some(&v) = self.str_intern.get(&ptr) {
            return v;
        }
        let v = self.alloc(Value::StrLit { ptr });
        self.str_intern.insert(ptr, v);
        v
    }

    pub(crate) fn mk_thunk(&mut self, env: VEnvId, expr: ExprPtr<'t>) -> ValId {
        if let Some(&v) = self.thunk_intern.get(&(env, expr)) {
            return v;
        }
        let v = self.alloc(Value::Thunk { env, expr, forced: None });
        self.thunk_intern.insert((env, expr), v);
        v
    }

    /// Record what a thunk evaluated to, so it is evaluated once.
    pub(crate) fn set_forced(&mut self, v: ValId, r: ValId) {
        match &mut self.vals[v as usize] {
            Value::Thunk { forced, .. } | Value::Unfold { forced, .. } => *forced = Some(r),
            _ => {}
        }
    }

    /// Record the value of a lambda's domain, so it is evaluated once.
    pub(crate) fn set_domain(&mut self, v: ValId, d: ValId) {
        if let Value::Lam { domain, .. } = &mut self.vals[v as usize] {
            *domain = Some(d);
        }
    }
}
