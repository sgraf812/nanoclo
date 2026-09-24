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
use crate::rc::{self, S, V};
use crate::util::{
    new_fx_hash_map, new_fx_hash_set, BigUintPtr, ExprPtr, FxHashMap, FxHashSet, LevelPtr,
    LevelsPtr, NamePtr, StringPtr,
};

pub(crate) type ValId = u32;
/// Environments are the checker's interned environments; evaluation extends
/// them through `push_entry` like everything else.
pub(crate) type VEnvId = crate::closure::EnvId;
pub(crate) type SpineId = u32;

/// The empty environment.
pub(crate) const VENV_NIL: VEnvId = crate::closure::ENV_NIL;
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

pub(crate) struct SpineNode<'t> {
    pub(crate) elim: Elim<'t>,
    pub(crate) parent: SpineId,
    pub(crate) len: u32,
}

/// The arenas, the tables that give equal constructions equal indices, and
/// the memos over them.
pub(crate) struct Vals<'t> {
    pub(crate) vals: crate::arena::Arena<Value<'t>>,
    pub(crate) spines: crate::arena::Arena<SpineNode<'t>>,

    // ---- interning ----
    /// `(parent, elimination) -> spine`
    /// applications keyed `(parent, value)`
    spine_intern_app: FxHashMap<(SpineId, ValId), SpineId>,
    spine_intern_proj: FxHashMap<(SpineId, NamePtr<'t>, usize), SpineId>,
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
    /// `(expression, environment) -> its value`, populated only at def-eq
    /// entry points
    pub(crate) clo_val_cache: FxHashMap<(ExprPtr<'t>, VEnvId), V>,
    /// `(constant, levels) -> the value of its body`
    pub(crate) unfold_cache: FxHashMap<(NamePtr<'t>, LevelsPtr<'t>), Option<V>>,
    /// `(constant, levels) -> the value denoting it`
    pub(crate) const_val_cache: FxHashMap<(NamePtr<'t>, LevelsPtr<'t>), V>,
    /// `(constant, levels) -> the value of its type`
    pub(crate) const_ty_cache: FxHashMap<(NamePtr<'t>, LevelsPtr<'t>), V>,
    /// `(constant, levels) -> the sort its type ends in`
    pub(crate) const_lvl_cache: FxHashMap<(NamePtr<'t>, LevelsPtr<'t>), Option<LevelPtr<'t>>>,
    /// `(recursor rule body, levels) -> the value of that body`
    pub(crate) rec_rule_cache: FxHashMap<(ExprPtr<'t>, LevelsPtr<'t>), V>,
    /// `stuck application -> what it reduces to`, `None` when it is stuck
    pub(crate) iota_cache: FxHashMap<ValId, Option<ValId>>,
    /// `(value, inductive) -> the constructor application it expands to`
    pub(crate) struct_eta_cache: FxHashMap<(ValId, NamePtr<'t>), Option<ValId>>,
    /// `value -> whether an opened binder or free variable occurs in it`
    pub(crate) open_cache: FxHashMap<ValId, bool>,
    /// `value -> the value of its type`
    pub(crate) type_cache: FxHashMap<ValId, ValId>,
    /// `local expression -> the value denoting it`
    pub(crate) local_cache: FxHashMap<ExprPtr<'t>, V>,
    /// `rapier environment -> the same bindings as values`

    // ---- conversion results ----
    pub(crate) conv_pos: FxHashSet<(ValId, ValId)>,
    pub(crate) conv_neg: FxHashSet<(ValId, ValId)>,
    /// Failures recorded while comparing the arguments of two applications of
    /// the same constant. That comparison is a guess that may fail while the
    /// terms are still equal, so its failures are dropped when the guess is.
    pub(crate) conv_neg_probe: FxHashSet<(ValId, ValId)>,
    /// Spine pairs whose speculative comparison settled false or exhausted
    /// its budget. A pair in this set goes straight to the unfolding route,
    /// so a mistaken bet pays its budget once.
    pub(crate) probe_fail: FxHashSet<(SpineId, SpineId)>,
    pub(crate) probe_depth: u32,
    /// Conversion steps the running speculation may still spend; one budget
    /// covers a comparison and everything it nests.
    pub(crate) probe_fuel: u64,
    /// Fuel granted to the next probe of an abort's own continuation: an
    /// aborted probe hands its doubled grant one step down the unfold
    /// chain, and the grant returns to the base once the chain resolves.
    pub(crate) probe_escalate: u64,
    pub(crate) probe_aborted: bool,
    pub(crate) in_conv: u32,
    /// Whether definitions unfold to their fused values (`fuse.rs`). Fixed
    /// for one attempt at a declaration.
    pub(crate) fuse: bool,
    /// Recursion wrapper unfoldings in this attempt, counted while `fuse` is
    /// off.
    pub(crate) wrap_count: u64,
}

impl<'t> Vals<'t> {
    pub(crate) fn new() -> Self {
        // The interning tables are built once per checking thread and kept
        // across declarations, so starting them at the size a mid-sized
        // declaration reaches spends one allocation instead of a rehash per
        // doubling on the way there.
        const PRE: usize = 1 << 14;
        fn pre<K: std::hash::Hash + Eq, V>() -> FxHashMap<K, V> {
            FxHashMap::with_capacity_and_hasher(PRE, Default::default())
        }
        Vals {
            vals: crate::arena::Arena::new(),
            // index 0 of each of these arenas is the empty case, so that
            // VENV_NIL and SPINE_EMPTY are valid indices needing no special
            // casing on the lookup paths.
            spines: {
                let mut s = crate::arena::Arena::new();
                rc::new_sentinel(rc::Kind::Spine);
                s.push(SpineNode {
                    elim: Elim::Proj {
                        ty_name: crate::util::Ptr::from(crate::util::DagMarker::ExportFile, 0),
                        idx: 0,
                    },
                    parent: 0,
                    len: 0,
                });
                s
            },
            spine_intern_app: pre(),
            spine_intern_proj: new_fx_hash_map(),
            rigid_intern: pre(),
            unfold_intern: pre(),
            lam_intern: pre(),
            pi_intern: new_fx_hash_map(),
            sort_intern: new_fx_hash_map(),
            nat_intern: new_fx_hash_map(),
            str_intern: new_fx_hash_map(),
            thunk_intern: pre(),
            clo_val_cache: pre(),
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
            conv_pos: new_fx_hash_set(),
            conv_neg: new_fx_hash_set(),
            conv_neg_probe: new_fx_hash_set(),
            probe_fail: new_fx_hash_set(),
            probe_depth: 0,
            probe_fuel: 0,
            probe_escalate: 0,
            probe_aborted: false,
            in_conv: 0,
            fuse: false,
            wrap_count: 0,
        }
    }

    /// Drop everything: values name arena positions, so nothing survives a
    /// declaration boundary. The arenas release their pages; a table keeps
    /// its capacity where it is not extravagant.
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
        rm(&mut self.spine_intern_app);
        rm(&mut self.spine_intern_proj);
        rm(&mut self.rigid_intern);
        rm(&mut self.unfold_intern);
        rm(&mut self.lam_intern);
        rm(&mut self.pi_intern);
        rm(&mut self.sort_intern);
        rm(&mut self.nat_intern);
        rm(&mut self.str_intern);
        rm(&mut self.thunk_intern);
        rm(&mut self.clo_val_cache);
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
        rs(&mut self.conv_pos);
        rs(&mut self.conv_neg);
        rs(&mut self.conv_neg_probe);
        rs(&mut self.probe_fail);
        self.probe_depth = 0;
        self.probe_fuel = 0;
        self.probe_escalate = 0;
        self.probe_aborted = false;
        self.wrap_count = 0;
        // the caches above held references; the nodes go only after them
        self.vals.truncate(0);
        self.spines.truncate(1);
        rc::reset(rc::Kind::Val, 0);
        rc::reset(rc::Kind::Spine, 1);
    }

    #[inline]
    pub(crate) fn get(&self, v: ValId) -> Value<'t> { self.vals[v as usize] }

    /// Append a value, taking a reference to each of its children.
    fn alloc(vals: &mut crate::arena::Arena<Value<'t>>, v: Value<'t>) -> ValId {
        let id = u32::try_from(vals.len()).expect("value arena overflow");
        match v {
            Value::Rigid { head, spine } => {
                if let RigidHead::BVar(_, ty) = head {
                    rc::inc_val(ty);
                }
                rc::inc_spine(spine);
            }
            Value::Unfold { spine, .. } => rc::inc_spine(spine),
            Value::Lam { env, .. } | Value::Thunk { env, .. } => rc::inc_env(env),
            Value::Pi { domain, env, .. } => {
                rc::inc_val(domain);
                rc::inc_env(env);
            }
            Value::Sort { .. } | Value::NatLit { .. } | Value::StrLit { .. } => {}
        }
        vals.push(v);
        rc::new_node(rc::Kind::Val);
        id
    }

    /// Append a spine node, taking a reference to its parent and argument.
    fn alloc_spine(spines: &mut crate::arena::Arena<SpineNode<'t>>, parent: SpineId, elim: Elim<'t>) -> SpineId {
        let len = spines[parent as usize].len + 1;
        let id = u32::try_from(spines.len()).expect("spine arena overflow");
        rc::inc_spine(parent);
        if let Elim::App(a) = elim {
            rc::inc_val(a);
        }
        spines.push(SpineNode { elim, parent, len });
        rc::new_node(rc::Kind::Spine);
        id
    }

    pub(crate) fn spine_len(&self, s: SpineId) -> u32 { self.spines[s as usize].len }

    /// Extend a spine. Interned, so a spine built twice the same way is the
    /// same spine.
    pub(crate) fn spine_snoc(&mut self, parent: SpineId, elim: Elim<'t>) -> S {
        let Vals { spines, spine_intern_app, spine_intern_proj, .. } = self;
        let id = match elim {
            Elim::App(a) => rc::intern(spine_intern_app, rc::Kind::Spine, (parent, a), || {
                Self::alloc_spine(spines, parent, elim)
            }),
            Elim::Proj { ty_name, idx } => {
                rc::intern(spine_intern_proj, rc::Kind::Spine, (parent, ty_name, idx), || {
                    Self::alloc_spine(spines, parent, elim)
                })
            }
        };
        S::own(id)
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

    pub(crate) fn mk_rigid(&mut self, head: RigidHead<'t>, spine: SpineId) -> V {
        let Vals { vals, rigid_intern, .. } = self;
        let id = rc::intern(rigid_intern, rc::Kind::Val, (head, spine), || Self::alloc(vals, Value::Rigid { head, spine }));
        V::own(id)
    }

    /// The variable standing for a binder opened at `level`. Interned on the
    /// level and the type, so two openings of the same binder are the same
    /// variable and everything built over them coincides.
    pub(crate) fn mk_bvar(&mut self, level: u32, ty: ValId) -> V {
        self.mk_rigid(RigidHead::BVar(level, ty), SPINE_EMPTY)
    }

    pub(crate) fn mk_unfold(
        &mut self,
        name: NamePtr<'t>,
        levels: LevelsPtr<'t>,
        spine: SpineId,
    ) -> V {
        let Vals { vals, unfold_intern, .. } = self;
        let id = rc::intern(unfold_intern, rc::Kind::Val, (name, levels, spine), || Self::alloc(vals, Value::Unfold { name, levels, spine, forced: None }));
        V::own(id)
    }

    pub(crate) fn mk_lam(
        &mut self,
        binder_name: NamePtr<'t>,
        binder_style: BinderStyle,
        binder_type: ExprPtr<'t>,
        env: VEnvId,
        body: ExprPtr<'t>,
    ) -> V {
        let Vals { vals, lam_intern, .. } = self;
        let id = rc::intern(lam_intern, rc::Kind::Val, (binder_type, env, body), || {
            Self::alloc(
                vals,
                Value::Lam { binder_name, binder_style, binder_type, domain: None, env, body },
            )
        });
        V::own(id)
    }

    pub(crate) fn mk_pi(
        &mut self,
        binder_name: NamePtr<'t>,
        binder_style: BinderStyle,
        domain: ValId,
        env: VEnvId,
        body: ExprPtr<'t>,
    ) -> V {
        let Vals { vals, pi_intern, .. } = self;
        let id = rc::intern(pi_intern, rc::Kind::Val, (domain, env, body), || Self::alloc(vals, Value::Pi { binder_name, binder_style, domain, env, body }));
        V::own(id)
    }

    pub(crate) fn mk_sort(&mut self, level: LevelPtr<'t>) -> V {
        let Vals { vals, sort_intern, .. } = self;
        V::own(rc::intern(sort_intern, rc::Kind::Val, level, || Self::alloc(vals, Value::Sort { level })))
    }

    pub(crate) fn mk_nat(&mut self, ptr: BigUintPtr<'t>) -> V {
        let Vals { vals, nat_intern, .. } = self;
        V::own(rc::intern(nat_intern, rc::Kind::Val, ptr, || Self::alloc(vals, Value::NatLit { ptr })))
    }

    pub(crate) fn mk_str(&mut self, ptr: StringPtr<'t>) -> V {
        let Vals { vals, str_intern, .. } = self;
        V::own(rc::intern(str_intern, rc::Kind::Val, ptr, || Self::alloc(vals, Value::StrLit { ptr })))
    }

    /// A thunk interned under `key_env`, the environment projected onto the
    /// entries `expr` reads, and forced under `env`, the environment whose
    /// indices `expr`'s variables name. Two closures agreeing on the read
    /// entries share the thunk, and with it the forced cell.
    pub(crate) fn mk_thunk_keyed(
        &mut self,
        key_env: VEnvId,
        env: VEnvId,
        expr: ExprPtr<'t>,
    ) -> V {
        let Vals { vals, thunk_intern, .. } = self;
        let id = rc::intern(thunk_intern, rc::Kind::Val, (key_env, expr), || Self::alloc(vals, Value::Thunk { env, expr, forced: None }));
        V::own(id)
    }

    /// Record what a thunk or a folded constant evaluated to, so it is
    /// evaluated once. The cell holds a reference to `r`, except when `r` is
    /// `v` itself, the mark of a constant that does not unfold.
    pub(crate) fn set_forced(&mut self, v: ValId, r: ValId) {
        let old = match &mut self.vals[v as usize] {
            Value::Thunk { forced, .. } | Value::Unfold { forced, .. } => forced.replace(r),
            _ => return,
        };
        if r != v {
            rc::inc_val(r);
        }
        if let Some(o) = old.filter(|&o| o != v) {
            rc::dec_val(o);
        }
    }

    /// Record the value of a lambda's domain, so it is evaluated once. The
    /// lambda holds a reference to it.
    pub(crate) fn set_domain(&mut self, v: ValId, d: ValId) {
        if let Value::Lam { domain, .. } = &mut self.vals[v as usize] {
            let old = domain.replace(d);
            rc::inc_val(d);
            if let Some(o) = old {
                rc::dec_val(o);
            }
        }
    }
}
