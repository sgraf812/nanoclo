/-
The same ladder built from `let` rather than beta: n nested `let x := 0`
with a body mentioning every binding.

Same origin as `beta-ladder.lean`: a symbolic execution trace names each
intermediate state, so `let s₁ := step s₀; let s₂ := step s₁; ...` is the
literal shape a trace takes before it is folded. Elaborated proof terms that
share subterms through `let` land here as well.

`infer_let` substitutes the value into the body eagerly in kernels that do
not delay substitution, giving the same quadratic blowup as the beta ladder;
a checker with environments records the binding and stays linear.

Measured at n = 16000 (arena config, single thread):
  rapier      0.025 s        ~n^1.13
  nanobruijn  96.3 s         ~n^2.44, 16.0 GB peak
  nanoda      14.5 s at n = 8000 (nanobruijn is slower still there: 17.2 s,
              its lazy-shift bookkeeping is pure overhead on this path)
-/
import Lean
open Lean Elab Command
set_option maxRecDepth 1000000
set_option maxHeartbeats 0
set_option debug.skipKernelTC true

run_elab do
  let nat := mkConst ``Nat
  let n := 4000
  let mut body : Expr := mkNatLit 0
  for i in [:n] do
    body := mkApp2 (mkConst ``Nat.add) (.bvar i) body
  for _ in [:n] do
    body := .letE `x nat (mkNatLit 0) body false
  Lean.addDecl (.defnDecl {
    name := `letLadder
    levelParams := []
    type := nat
    value := body
    hints := .regular 0
    safety := .safe
  })
