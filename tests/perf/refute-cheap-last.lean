/-
Mechanism: the order in which the arguments of a constructor application are
visited while refuting it, with the cheap refutation in the last argument.

Test case from Courant and Leroy, "A Lazy, Concurrent Convertibility Checker"
(POPL 2026), section 10, where the two argument orders cost Rocq 4 × 10⁻⁶s and
0.61s.

Write `#k` for the unary numeral `N.S (N.S (... N.O))` with k successors.
`count #k` evaluates to `#k` in Θ(k²) reductions, because each of the k steps
of the recursion adds one to the numeral accumulated so far and `N.add`
recurses over its first argument.

The declaration compares

  dropArg (count #n, false)   with   dropArg (count #(n+1), true)

The two pairs are not convertible, and each of the two components refutes them
on its own: the first components are numerals of different value, at a cost of
Θ(n²), and the second components are `false` against `true`, at a cost of
Θ(1). The cheap refutation is the last argument of `Prod.mk`. Once the pairs
have been refuted, unfolding `dropArg` on both sides leaves `N.O` against
`N.O`, which is why the declaration holds.

Paired with `refute-cheap-first.lean`, which carries the same two components in
the opposite positions. The ratio between the two files is the cost an engine
pays for the order in which it visits arguments.

`N.add` and `count` are written with `N.rec` applied directly, so each exported
definition is a single recursor application, with no `WellFounded.fix` or
`brecOn` wrapper in between.

The recursion depth reached is Θ(n).

For lean-kernel-arena:
  lean4export refute-cheap-last.lean > test.jsonl
  # declaration to check: kernel_refute_cheap_last
-/
import Lean

open Lean Elab Command

set_option maxRecDepth 1000000
set_option maxHeartbeats 0
set_option debug.skipKernelTC true

inductive N where
  | O : N
  | S : N → N

noncomputable def N.add (a b : N) : N :=
  N.rec (motive := fun _ => N) b (fun _ ih => N.S ih) a

noncomputable def count (m : N) : N :=
  N.rec (motive := fun _ => N) N.O (fun _ ih => N.add ih (N.S N.O)) m

def dropArg (_ : N × Bool) : N := N.O

run_elab do
  let n := 200
  let ty := mkConst ``N
  let num : Nat → Expr := fun k => Id.run do
    let mut e := mkConst ``N.O
    for _ in [:k] do
      e := mkApp (mkConst ``N.S) e
    return e
  let pair : Expr → Expr → Expr := fun a b =>
    mkApp4 (mkConst ``Prod.mk [0, 0]) ty (mkConst ``Bool) a b
  let lhs := mkApp (mkConst ``dropArg)
    (pair (mkApp (mkConst ``count) (num n)) (mkConst ``Bool.false))
  let rhs := mkApp (mkConst ``dropArg)
    (pair (mkApp (mkConst ``count) (num (n + 1))) (mkConst ``Bool.true))
  Lean.addDecl (.thmDecl {
    name := `kernel_refute_cheap_last
    levelParams := []
    type := mkApp3 (mkConst ``Eq [1]) ty lhs rhs
    value := mkApp2 (mkConst ``Eq.refl [1]) ty lhs
  })
