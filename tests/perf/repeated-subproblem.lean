/-
Mechanism: reaching the same convertibility subproblem an exponential number of
times, so that the cost depends on whether convertibility results are shared.

Test case from Courant and Leroy, "A Lazy, Concurrent Convertibility Checker"
(POPL 2026), section 10, where Rocq spends 0.018s and the paper's checker
spends 9 × 10⁻⁵s.

Write `#k` for the unary numeral `N.S (N.S (... N.O))` with k successors.
`perfect #k t` builds the perfect binary tree of depth k whose leaves are `t`.
Its k reduction steps each duplicate the tree built so far into both arguments
of `Tr.node`, so the result has 2^k leaves and a representation of size k.

The declaration compares

  perfect #n leaf   with   perfect #(n-1) (node leaf leaf)

Both sides reduce to the perfect tree of depth n. Comparing them descends
through `Tr.node u u` against `Tr.node v v` at every level, and the pair (u, v)
reached from the left argument is the pair reached from the right argument, so
the recursion visits 2^n pairs of nodes while only n of them are distinct. An
engine that records which pairs of terms it has already proved convertible
answers in Θ(n); an engine that solves each occurrence afresh answers in
Θ(2^n). Stepping n by one doubles the cost of the second behaviour and adds a
constant to the cost of the first.

`perfect` is written with `N.rec` applied directly, so the exported definition
is a single recursor application, with no `WellFounded.fix` or `brecOn` wrapper
in between.

For lean-kernel-arena:
  lean4export repeated-subproblem.lean > test.jsonl
  # declaration to check: kernel_repeated_subproblem
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

noncomputable def perfect (k : N) (t : Tr) : Tr :=
  N.rec (motive := fun _ => Tr → Tr) (fun t => t) (fun _ ih => fun t => ih (Tr.node t t)) k t

run_elab do
  let n := 20
  let ty := mkConst ``Tr
  let leaf := mkConst ``Tr.leaf
  let num : Nat → Expr := fun k => Id.run do
    let mut e := mkConst ``N.O
    for _ in [:k] do
      e := mkApp (mkConst ``N.S) e
    return e
  let lhs := mkApp2 (mkConst ``perfect) (num n) leaf
  let rhs := mkApp2 (mkConst ``perfect) (num (n - 1)) (mkApp2 (mkConst ``Tr.node) leaf leaf)
  Lean.addDecl (.thmDecl {
    name := `kernel_repeated_subproblem
    levelParams := []
    type := mkApp3 (mkConst ``Eq [1]) ty lhs rhs
    value := mkApp2 (mkConst ``Eq.refl [1]) ty lhs
  })
