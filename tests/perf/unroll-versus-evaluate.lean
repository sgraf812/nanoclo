/-
Mechanism: choosing which of two different head constants to unfold, where one
unrolling step of the recursive side reproduces the other side.

Test case from Courant and Leroy, "A Lazy, Concurrent Convertibility Checker"
(POPL 2026), section 2, which poses `exp 40 ≈ exp 39 + exp 39` and the
mutually recursive `odd 999999 ≈ even 1000000`.

Write `#k` for the unary numeral `N.S (N.S (... N.O))` with k successors.
`count #k` evaluates to `#k` in Θ(k²) reductions, because each of the k steps
of the recursion adds one to the numeral accumulated so far and `N.add`
recurses over its first argument.

The declaration compares

  count #(n+1)   with   N.add (count #n) #1

The head constants are `count` on the left and `N.add` on the right. Unrolling
`count` by one step turns the left side into `N.add (count #n) #1`, which is
the right side, so the problem is settled by that one step together with a
traversal of the shared numeral `#n`, in Θ(n). Evaluating the two sides instead
costs Θ(n²). Running at n, 2n, 4n separates the two behaviours as an exponent
of 1 against an exponent of 2.

`N.add` and `count` are written with `N.rec` applied directly, so each exported
definition is a single recursor application, with no `WellFounded.fix` or
`brecOn` wrapper in between.

The recursion depth reached is Θ(n).

For lean-kernel-arena:
  lean4export unroll-versus-evaluate.lean > test.jsonl
  # declaration to check: kernel_unroll_versus_evaluate
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
  let numN := num n
  let one := mkApp (mkConst ``N.S) (mkConst ``N.O)
  let lhs := mkApp (mkConst ``count) (mkApp (mkConst ``N.S) numN)
  let rhs := mkApp2 (mkConst ``N.add) (mkApp (mkConst ``count) numN) one
  Lean.addDecl (.thmDecl {
    name := `kernel_unroll_versus_evaluate
    levelParams := []
    type := mkApp3 (mkConst ``Eq [1]) ty lhs rhs
    value := mkApp2 (mkConst ``Eq.refl [1]) ty lhs
  })
