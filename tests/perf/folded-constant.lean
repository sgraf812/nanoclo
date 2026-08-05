/-
Mechanism: recognizing the folded form of a constant application after a shared
occurrence of it has been reduced to weak head normal form for an earlier
subproblem.

Test case from Courant and Leroy, "A Lazy, Concurrent Convertibility Checker"
(POPL 2026), section 10, where Rocq spends 0.078s and the paper's checker
spends 2 × 10⁻⁴s.

Write `#k` for the unary numeral `N.S (N.S (... N.O))` with k successors.
`count #k` evaluates to `#k` in Θ(k²) reductions, because each of the k steps
of the recursion adds one to the numeral accumulated so far and `N.add`
recurses over its first argument. Reaching the head constructor of `count #k`
alone takes Θ(k) reductions. `tagged m` pairs `m` with the test `isZero m`, so
unfolding `tagged` duplicates its argument into both components of the pair.

The declaration compares

  tagged (count #n)   with   (count #n, false)

Unfolding `tagged` on the left gives `(count #n, isZero (count #n))`, where the
two occurrences of `count #n` are one shared subterm. The second components,
`isZero (count #n)` against `false`, are settled by reducing that shared
subterm to its head constructor, in Θ(n). The first components are then
`count #n` against `count #n`. An engine for which the shared occurrence still
carries its folded form settles them by identity, for Θ(n) overall; an engine
for which the shared occurrence now holds the weak head normal form evaluates
both numerals, for Θ(n²) overall. Running at n, 2n, 4n separates the two
behaviours as an exponent of 1 against an exponent of 2.

`N.add`, `count` and `isZero` are written with `N.rec` applied directly, so
each exported definition is a single recursor application, with no
`WellFounded.fix` or `brecOn` wrapper in between.

The recursion depth reached is Θ(n).

For lean-kernel-arena:
  lean4export folded-constant.lean > test.jsonl
  # declaration to check: kernel_folded_constant
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

noncomputable def isZero (m : N) : Bool :=
  N.rec (motive := fun _ => Bool) true (fun _ _ => false) m

noncomputable def tagged (m : N) : N × Bool := (m, isZero m)

run_elab do
  let n := 1000
  let ty := mkConst ``N
  let pairTy := mkApp2 (mkConst ``Prod [0, 0]) ty (mkConst ``Bool)
  let num : Nat → Expr := fun k => Id.run do
    let mut e := mkConst ``N.O
    for _ in [:k] do
      e := mkApp (mkConst ``N.S) e
    return e
  let arg := mkApp (mkConst ``count) (num n)
  let lhs := mkApp (mkConst ``tagged) arg
  let rhs := mkApp4 (mkConst ``Prod.mk [0, 0]) ty (mkConst ``Bool) arg (mkConst ``Bool.false)
  Lean.addDecl (.thmDecl {
    name := `kernel_folded_constant
    levelParams := []
    type := mkApp3 (mkConst ``Eq [1]) pairTy lhs rhs
    value := mkApp2 (mkConst ``Eq.refl [1]) pairTy lhs
  })
