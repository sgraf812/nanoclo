/-
Reproducer for O(n²) kernel type-checking. Nested `let` bindings.
From kraken, same family as kernel-congr-quadratic-mwe.lean.

Structure (for n=3):
  let x₃ : Nat := 0
  let x₂ : Nat := 0
  let x₁ : Nat := 0
  x₃ + (x₂ + (x₁ + 0))

The body of each `let` still contains the next one, and the innermost sum
references every binding, so each body has `loose_bvar_range > 0`.

`infer_let` substitutes the value into the body before checking it, and the
body has size O(n) and contains the next `let`, so the kernel performs n
substitutions each traversing O(n) nodes = O(n²) total. A checker that
records the binding in an environment stays linear. This is the zeta
counterpart of beta-ladder.lean; no application heads are involved, so it
isolates `let` processing from beta reduction.

For lean-kernel-arena:
  lean4export let-ladder.lean > test.jsonl
  # declaration to check: kernel_quadratic_let_ladder
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
  -- Wrap in nested lets
  for _ in [:n] do
    body := .letE `x nat (mkNatLit 0) body false
  Lean.addDecl (.defnDecl {
    name := `kernel_quadratic_let_ladder
    levelParams := []
    type := nat
    value := body
    hints := .regular 0
    safety := .safe
  })
