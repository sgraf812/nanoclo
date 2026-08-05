/-
Mechanism: beta reduction under binders, on two terms whose normal forms are
long and carry no repeated subterms.

Test case from the `conv_eval` benchmark of András Kovács' smalltt, whose
`NatConv` entry compares Church numerals of a given value.

`cnum k` is the Church numeral `fun X s z => s (s (... z))` with k applications
of `s`, written out as a term of size k. `cmul a b` iterates `b` as many times
as `a` counts.

The declaration compares

  cmul (cnum n) (cnum (n+1))   with   cmul (cnum (n+1)) (cnum n)

Both sides have the normal form `fun X s z => s (s (... z))` with n(n+1)
applications of the free variable `s`. Unfolding `cmul` on the left leaves
`cnum n X (cnum (n+1) X s)` under the binders `X`, `s`, `z`, and each of the n
occurrences of the bound function in `cnum n` is a separate copy of the redex
`cnum (n+1) X s`, so reaching the normal form takes n(n+1) beta steps and
allocates a term with no repeated subterms. The exported terms have size Θ(n)
and the decision costs Θ(n²), so running at n, 2n, 4n gives an exponent of 2.
The only delta step available is the unfolding of `cmul` on each side, and the
only argument comparison available, `cnum n` against `cnum (n+1)`, costs Θ(n),
so the figure measures the rate at which an engine performs beta reduction
under binders.

The recursion depth reached is n(n+1), which is 14520 at the default n.

For lean-kernel-arena:
  lean4export church-numerals.lean > test.jsonl
  # declaration to check: kernel_church_numerals
-/
import Lean

open Lean Elab Command

set_option maxRecDepth 1000000
set_option maxHeartbeats 0
set_option debug.skipKernelTC true

def CNat : Type 1 := ∀ (X : Type), (X → X) → X → X

def cmul (a b : CNat) : CNat := fun X s => a X (b X s)

run_elab do
  let n := 120
  let ty := mkConst ``CNat
  -- `fun (X : Type) (s : X → X) (z : X) => s (s (... z))`, k applications of `s`
  let cnum : Nat → Expr := fun k => Id.run do
    let mut body : Expr := .bvar 0
    for _ in [:k] do
      body := mkApp (.bvar 1) body
    let tyX : Expr := .sort (.succ .zero)
    let tyS : Expr := .forallE `x (.bvar 0) (.bvar 1) .default
    return .lam `X tyX (.lam `s tyS (.lam `z (.bvar 1) body .default) .default) .default
  let lhs := mkApp2 (mkConst ``cmul) (cnum n) (cnum (n + 1))
  let rhs := mkApp2 (mkConst ``cmul) (cnum (n + 1)) (cnum n)
  Lean.addDecl (.thmDecl {
    name := `kernel_church_numerals
    levelParams := []
    type := mkApp3 (mkConst ``Eq [2]) ty lhs rhs
    value := mkApp2 (mkConst ``Eq.refl [2]) ty lhs
  })
