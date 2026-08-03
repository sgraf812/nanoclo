/-
Reproducer for O(n²) kernel type-checking. Nested `let` bindings.
From kraken.

Structure (for n=3):
  let x₃ : Nat := 0
  let x₂ : Nat := 0
  let x₁ : Nat := 0
  x₃ + (x₂ + (x₁ + 0))

The body of each `let` still contains the next one, and the innermost sum
references every binding, so each body has `loose_bvar_range > 0`. No
application heads are involved beyond `Nat.add`, so this exercises `let`
processing on its own, with no beta reduction.

A kernel that substitutes the value into the body before checking it
traverses a body of size O(n) at each of the n bindings, and the result
contains the next `let`, giving O(n²) total in time and in allocated
nodes. Recording the binding and looking it up on demand makes each step
O(1) and the whole term linear.

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
