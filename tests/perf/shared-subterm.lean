/-
Mechanism: consuming a term whose normal form has 2^n nodes and whose
representation has n nodes, without expanding the representation.

Test case from Courant and Leroy, "A Lazy, Concurrent Convertibility Checker"
(POPL 2026), section 10.

Write `#k` for the unary numeral `N.S (N.S (... N.O))` with k successors.
`perfect #k leaf` builds the perfect binary tree of depth k. Its k reduction
steps each duplicate the tree built so far into both arguments of `Tr.node`, so
the result has 2^k leaves and a representation of size k. `ldepth` and
`ldepth2` both return the length of the leftmost path of a tree, `ldepth` by
placing one `N.S` per level and `ldepth2` by adding one to the count obtained
from the level below.

The declaration compares

  ldepth (perfect #n leaf)   with   ldepth2 (perfect #n leaf)

Both sides evaluate to `#n`. The two head constants differ, so both sides are
evaluated: `ldepth` walks the n nodes of the leftmost path for Θ(n), `ldepth2`
walks the same path and folds n additions over numerals of growing size for
Θ(n²), and comparing the two resulting numerals costs Θ(n). Neither traversal
looks at more than the leftmost path, so an engine that keeps the duplicated
subtrees shared answers in Θ(n²), while an engine that expands `perfect #n
leaf` into its normal form allocates 2^n nodes. Running at n, 2n, 4n gives an
exponent of 2 for the first behaviour.

`perfect`, `ldepth` and `ldepth2` are written with `N.rec` and `Tr.rec` applied
directly, so each exported definition is a single recursor application, with no
`WellFounded.fix` or `brecOn` wrapper in between.

The recursion depth reached is Θ(n).

For lean-kernel-arena:
  lean4export shared-subterm.lean > test.jsonl
  # declaration to check: kernel_shared_subterm
-/
import Lean

open Lean Elab Command

set_option maxRecDepth 1000000
set_option maxHeartbeats 0
set_option debug.skipKernelTC true

inductive N where
  | O : N
  | S : N → N

inductive Tr where
  | leaf : Tr
  | node : Tr → Tr → Tr

noncomputable def N.add (a b : N) : N :=
  N.rec (motive := fun _ => N) b (fun _ ih => N.S ih) a

noncomputable def perfect (k : N) (t : Tr) : Tr :=
  N.rec (motive := fun _ => Tr → Tr) (fun t => t) (fun _ ih => fun t => ih (Tr.node t t)) k t

noncomputable def ldepth (t : Tr) : N :=
  Tr.rec (motive := fun _ => N) N.O (fun _ _ ih _ => N.S ih) t

noncomputable def ldepth2 (t : Tr) : N :=
  Tr.rec (motive := fun _ => N) N.O (fun _ _ ih _ => N.add ih (N.S N.O)) t

run_elab do
  let n := 1000
  let ty := mkConst ``N
  let num : Nat → Expr := fun k => Id.run do
    let mut e := mkConst ``N.O
    for _ in [:k] do
      e := mkApp (mkConst ``N.S) e
    return e
  let tree := mkApp2 (mkConst ``perfect) (num n) (mkConst ``Tr.leaf)
  let lhs := mkApp (mkConst ``ldepth) tree
  let rhs := mkApp (mkConst ``ldepth2) tree
  Lean.addDecl (.thmDecl {
    name := `kernel_shared_subterm
    levelParams := []
    type := mkApp3 (mkConst ``Eq [1]) ty lhs rhs
    value := mkApp2 (mkConst ``Eq.refl [1]) ty lhs
  })
