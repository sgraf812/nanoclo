/-
Reproducer for O(n²) kernel type-checking. Nested beta redexes.
From kraken, same family as kernel-congr-quadratic-mwe.lean.

Structure (for n=3):
  (fun x₃ : Nat =>
    (fun x₂ : Nat =>
      (fun x₁ : Nat => x₃ + (x₂ + (x₁ + 0))) 0
    ) 0
  ) 0

Every redex is a lambda applied to one argument, and the body of each
lambda still contains the next redex. The innermost sum references all
enclosing binders, so every continuation body has `loose_bvar_range > 0`.

Reducing the outer redex substitutes the argument into a body of size O(n),
and the result contains the next redex, so the kernel performs n
substitutions each traversing O(n) nodes = O(n²) total, in time and in
allocated nodes. A checker that records the argument in an environment
instead enters each binder in O(1) and stays linear.

Note that a *flat* spine `(fun x₁ ... xₙ => ...) a₁ ... aₙ` does not
reproduce this: kernels collect the whole spine and substitute once. The
redexes have to be nested.

For lean-kernel-arena:
  lean4export beta-ladder.lean > test.jsonl
  # declaration to check: kernel_quadratic_beta_ladder
-/
import Lean

open Lean Elab Command

set_option maxRecDepth 1000000
set_option maxHeartbeats 0
set_option debug.skipKernelTC true

run_elab do
  let nat := mkConst ``Nat
  let n := 4000
  -- Innermost body: xₙ + (xₙ₋₁ + ... + (x₁ + 0))
  let mut body : Expr := mkNatLit 0
  for i in [:n] do
    body := mkApp2 (mkConst ``Nat.add) (.bvar i) body
  -- Wrap in nested redexes, each lambda applied to 0
  for _ in [:n] do
    body := mkApp (.lam `x nat body .default) (mkNatLit 0)
  Lean.addDecl (.thmDecl {
    name := `kernel_quadratic_beta_ladder
    levelParams := []
    type := mkApp3 (mkConst ``Eq [1]) nat body (mkNatLit 0)
    value := mkApp2 (mkConst ``Eq.refl [1]) nat (mkNatLit 0)
  })
