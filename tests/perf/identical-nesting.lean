/-
Mechanism: answering a convertibility problem whose two sides are the same
deeply nested term, by observing that they are the same term.

Test case from Courant and Leroy, "A Lazy, Concurrent Convertibility Checker"
(POPL 2026), section 10, where Rocq spends 2 × 10⁻⁵s and the paper's checker
spends 0.15s, its largest loss against Rocq.

`f0` is the identity on `N` and each of `f1`, `f2`, `f3`, `f4` applies its
predecessor twice, so `f4` expands into 16 applications of `f0` and an n-fold
nesting of `f4` expands into 16n of them.

The declaration compares

  f4 (f4 (... (f4 N.O) ...))   with   f4 (f4 (... (f4 N.O) ...))

with n applications of `f4` on each side. The two sides are the same term, and
each side reduces to `N.O`. An engine that compares the two sides before
unfolding anything answers in Θ(n). An engine that unfolds one side at a time
has 16n applications to choose from on each side, and the number of pairs of
partially unfolded sides it can reach grows exponentially in n.

For lean-kernel-arena:
  lean4export identical-nesting.lean > test.jsonl
  # declaration to check: kernel_identical_nesting
-/
import Lean

open Lean Elab Command

set_option maxRecDepth 1000000
set_option maxHeartbeats 0
set_option debug.skipKernelTC true

inductive N where
  | O : N
  | S : N → N

def f0 (x : N) : N := x
def f1 (x : N) : N := f0 (f0 x)
def f2 (x : N) : N := f1 (f1 x)
def f3 (x : N) : N := f2 (f2 x)
def f4 (x : N) : N := f3 (f3 x)

run_elab do
  let n := 30
  let ty := mkConst ``N
  let mut nest : Expr := mkConst ``N.O
  for _ in [:n] do
    nest := mkApp (mkConst ``f4) nest
  Lean.addDecl (.thmDecl {
    name := `kernel_identical_nesting
    levelParams := []
    type := mkApp3 (mkConst ``Eq [1]) ty nest nest
    value := mkApp2 (mkConst ``Eq.refl [1]) ty nest
  })
