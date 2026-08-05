/-
Mechanism: deciding the arguments of two applications of the same constant
convertible before unfolding that constant.

Test case from Courant and Leroy, "A Lazy, Concurrent Convertibility Checker"
(POPL 2026), section 10.

Write `#k` for the unary numeral `N.S (N.S (... N.O))` with k successors.
`count #k` evaluates to `#k`. Each of the k steps of the recursion adds one to
the numeral accumulated so far, and `N.add` recurses over its first argument,
so evaluating `count #k` performs Θ(k²) reductions.

The declaration compares

  count #n   with   count (N.add #(n-1) #1)

`N.add #(n-1) #1` reduces to `#n` in n steps, so the two arguments of `count`
are convertible at cost Θ(n) and the two applications agree without `count`
being unfolded. Evaluating the two applications of `count` instead costs
Θ(n²). Running at n, 2n, 4n separates the two behaviours as an exponent of 1
against an exponent of 2.

`N.add` and `count` are written with `N.rec` applied directly, so each exported
definition is a single recursor application, with no `WellFounded.fix` or
`brecOn` wrapper in between.

The recursion depth reached is Θ(n).

For lean-kernel-arena:
  lean4export args-before-unfold.lean > test.jsonl
  # declaration to check: kernel_args_before_unfold
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

run_elab do
  let n := 1000
  let ty := mkConst ``N
  let num : Nat → Expr := fun k => Id.run do
    let mut e := mkConst ``N.O
    for _ in [:k] do
      e := mkApp (mkConst ``N.S) e
    return e
  let lhs := mkApp (mkConst ``count) (num n)
  let rhs := mkApp (mkConst ``count) (mkApp2 (mkConst ``N.add) (num (n - 1)) (num 1))
  Lean.addDecl (.thmDecl {
    name := `kernel_args_before_unfold
    levelParams := []
    type := mkApp3 (mkConst ``Eq [1]) ty lhs rhs
    value := mkApp2 (mkConst ``Eq.refl [1]) ty lhs
  })
