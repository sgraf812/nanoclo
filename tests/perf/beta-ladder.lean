/-
Nested beta redexes: `e_{i+1} = (fun x : Nat => e_i) 0`, n deep, where the
innermost body mentions every binder.

Where this shape comes from: a symbolic execution trace. Stepping an
interpreter or a machine semantics emits one binding per step whose value
mentions the state built by the previous steps, and folding the trace
produces exactly this ladder. Tactic output that beta-expands a continuation
per step (do-notation desugaring, `Function.comp` chains, instance
projections) produces it too.

A checker that substitutes when it enters a binder copies a body of size
O(n) at each of the n levels, so it is quadratic in time and space even
though the term is linear. A checker that binds the argument in an
environment enters each binder in O(1) and stays linear.

Measured at n = 16000 (arena config, single thread):
  rapier      0.045 s        ~n^1.07
  nanobruijn  123.8 s        ~n^2.40, 19.6 GB peak (4x per doubling)
  nanoda      exceeds 120 s at n = 8000

`debug.skipKernelTC` keeps Lean from checking the term as it is added, so
the export is produced without paying the cost under test.
-/
import Lean
open Lean Elab Command
set_option maxRecDepth 1000000
set_option maxHeartbeats 0
set_option debug.skipKernelTC true

run_elab do
  let nat := mkConst ``Nat
  let n := 4000
  -- body: Nat.add x_1 (Nat.add x_2 (... (Nat.add x_n 0)))
  let mut body : Expr := mkNatLit 0
  for i in [:n] do
    body := mkApp2 (mkConst ``Nat.add) (.bvar i) body
  -- ladder: (fun x_n => ... ((fun x_1 => body) 0) ...) 0
  for _ in [:n] do
    body := mkApp (.lam `x nat body .default) (mkNatLit 0)
  Lean.addDecl (.thmDecl {
    name := `betaLadder
    levelParams := []
    type := mkApp3 (mkConst ``Eq [1]) nat body (mkNatLit 0)
    value := mkApp2 (mkConst ``Eq.refl [1]) nat (mkNatLit 0)
  })
