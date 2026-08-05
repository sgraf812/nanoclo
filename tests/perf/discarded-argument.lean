/-
Mechanism: comparing the arguments of two applications of the same constant
when that constant discards its argument.

Test case from Courant and Leroy, "A Lazy, Concurrent Convertibility Checker"
(POPL 2026), section 10, where Rocq spends 0.14s and the paper's checker
spends 5 × 10⁻⁶s.

Write `#k` for the unary numeral `N.S (N.S (... N.O))` with k successors.
`count #k` evaluates to `#k` in Θ(k²) reductions, because each of the k steps
of the recursion adds one to the numeral accumulated so far and `N.add`
recurses over its first argument. `dropArg` maps every numeral to `N.O`.

The declaration compares

  dropArg (count #n)   with   dropArg (count #(n+1))

The two arguments of `dropArg` evaluate to different numerals, so establishing
that they are not convertible costs Θ(n²) and the answer is then thrown away:
unfolding `dropArg` on both sides leaves `N.O` against `N.O`. An engine that
unfolds `dropArg` before looking at its argument answers in Θ(1), and one that
compares arguments first answers in Θ(n²). Running at n, 2n, 4n separates the
two behaviours as an exponent of 0 against an exponent of 2.

`N.add` and `count` are written with `N.rec` applied directly, so each exported
definition is a single recursor application, with no `WellFounded.fix` or
`brecOn` wrapper in between.

The recursion depth reached is Θ(n).

For lean-kernel-arena:
  lean4export discarded-argument.lean > test.jsonl
  # declaration to check: kernel_discarded_argument
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

def dropArg (_ : N) : N := N.O

run_elab do
  let n := 200
  let ty := mkConst ``N
  let num : Nat → Expr := fun k => Id.run do
    let mut e := mkConst ``N.O
    for _ in [:k] do
      e := mkApp (mkConst ``N.S) e
    return e
  let lhs := mkApp (mkConst ``dropArg) (mkApp (mkConst ``count) (num n))
  let rhs := mkApp (mkConst ``dropArg) (mkApp (mkConst ``count) (num (n + 1)))
  Lean.addDecl (.thmDecl {
    name := `kernel_discarded_argument
    levelParams := []
    type := mkApp3 (mkConst ``Eq [1]) ty lhs rhs
    value := mkApp2 (mkConst ``Eq.refl [1]) ty lhs
  })
