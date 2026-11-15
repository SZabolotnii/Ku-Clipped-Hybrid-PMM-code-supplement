import Mathlib

/-!
# Data-driven choice of the basis parameter: the deterministic core

Statements behind `theory/t8_selector.tex` (track Ku-Clipped-Hybrid-PMM), drafted 2026-11-13
with `sorry` proofs and frozen by `lean_gate.py snapshot` before any proof was written.

Setting. A finite family of bases is indexed by `a : ι` (the grid of the exponent parameter
`α`). `J a = b'F⁻¹b` is the information of basis `a`, so `1 / J a` is its asymptotic variance up
to a common factor. `Jhat a` is the sample criterion and `ahat` a maximiser of `Jhat`.

What is formalised is everything in the selector theorem that is not probability:

* `selector_oracle`, `selector_variance_regret` — if the criterion is within a relative `ε` of
  `J` on the whole grid, the selected basis loses at most the factor `(1 + ε) / (1 - ε)`;
* `selector_oracle_bias_noise` — the same with the error split into a deterministic bias
  (`Jdelta` against `J`, from the smoothing width `δ`) and sampling noise;
* `selector_additive` — the additive form;
* `selector_consistent` — the selector picks the maximiser of the LIMIT of the criterion as
  soon as the sampling error is below half the gap;
* `selector_regret_of_biased_limit` — hence, when the limit criterion and the target variance
  `V` have different optimisers, the regret does not vanish however small the noise;
* `one_step_pilot_share` — a one-step estimator whose slope `c'` differs from the true slope `c`
  keeps the share `1 - c / c'` of the pilot error;
* `misweighted_variance_bound` — any weights `h` give variance `h'Fh / (h'b)²` at least
  `1 / (b'F⁻¹b)`.

Convergence in probability of `Jhat`, and asymptotic normality of the estimators, are proved on
paper (`theory/t5_one_step.tex`, `theory/t8_selector.tex`); they are not formalised.
-/

set_option autoImplicit false

open Matrix

namespace TauPATP

/-- Oracle inequality, information form. -/
theorem selector_oracle {ι : Type*} (J Jhat : ι → ℝ) (ε : ℝ) (ahat : ι)
    (hε0 : 0 ≤ ε) (hε1 : ε < 1) (hJ : ∀ a, 0 < J a)
    (hclose : ∀ a, |Jhat a - J a| ≤ ε * J a)
    (hmax : ∀ a, Jhat a ≤ Jhat ahat) :
    ∀ a, (1 - ε) / (1 + ε) * J a ≤ J ahat := by
  intro a
  have h1 := abs_le.mp (hclose a)
  have h2 := abs_le.mp (hclose ahat)
  have hm := hmax a
  have hpos : 0 < 1 + ε := by linarith
  rw [div_mul_eq_mul_div, div_le_iff₀ hpos]
  nlinarith [h1.1, h2.2, hJ a, hJ ahat]

theorem selector_oracle_witness :
    ∃ (J Jhat : Bool → ℝ) (ε : ℝ) (ahat : Bool),
      0 ≤ ε ∧ ε < 1 ∧ (∀ a, 0 < J a) ∧ (∀ a, |Jhat a - J a| ≤ ε * J a) ∧
        (∀ a, Jhat a ≤ Jhat ahat) := by
  refine ⟨fun a => if a then 2 else 1, fun a => if a then 2.1 else 0.95, 1 / 10, true,
    by norm_num, by norm_num, ?_, ?_, ?_⟩
  · intro a; cases a <;> norm_num
  · intro a; cases a <;> norm_num [abs_le]
  · intro a; cases a <;> norm_num

/-- Oracle inequality, variance form: the selected basis has asymptotic variance at most
`(1 + ε) / (1 - ε)` times that of any basis of the grid. -/
theorem selector_variance_regret {ι : Type*} (J Jhat : ι → ℝ) (ε : ℝ) (ahat : ι)
    (hε0 : 0 ≤ ε) (hε1 : ε < 1) (hJ : ∀ a, 0 < J a)
    (hclose : ∀ a, |Jhat a - J a| ≤ ε * J a)
    (hmax : ∀ a, Jhat a ≤ Jhat ahat) :
    ∀ a, 1 / J ahat ≤ (1 + ε) / (1 - ε) * (1 / J a) := by
  intro a
  have h1 := abs_le.mp (hclose a)
  have h2 := abs_le.mp (hclose ahat)
  have hm := hmax a
  have hden : 0 < (1 - ε) * J a := mul_pos (by linarith) (hJ a)
  rw [div_mul_div_comm, mul_one, div_le_div_iff₀ (hJ ahat) hden]
  nlinarith [h1.1, h2.2, hJ a, hJ ahat]

theorem selector_variance_regret_witness :
    ∃ (J Jhat : Bool → ℝ) (ε : ℝ) (ahat : Bool),
      0 ≤ ε ∧ ε < 1 ∧ (∀ a, 0 < J a) ∧ (∀ a, |Jhat a - J a| ≤ ε * J a) ∧
        (∀ a, Jhat a ≤ Jhat ahat) := by
  refine ⟨fun a => if a then 2 else 1, fun a => if a then 2.1 else 0.95, 1 / 10, true,
    by norm_num, by norm_num, ?_, ?_, ?_⟩
  · intro a; cases a <;> norm_num
  · intro a; cases a <;> norm_num [abs_le]
  · intro a; cases a <;> norm_num

/-- Oracle inequality with the error of the criterion split into a bias `εb` (the limit
`Jdelta` of the criterion against the information `J`) and sampling noise `εn`. -/
theorem selector_oracle_bias_noise {ι : Type*} (J Jdelta Jhat : ι → ℝ) (εb εn : ℝ) (ahat : ι)
    (hεb0 : 0 ≤ εb) (hεn0 : 0 ≤ εn) (hεn1 : εn ≤ 1) (hJ : ∀ a, 0 < J a)
    (hbias : ∀ a, |Jdelta a - J a| ≤ εb * J a)
    (hnoise : ∀ a, |Jhat a - Jdelta a| ≤ εn * Jdelta a)
    (hmax : ∀ a, Jhat a ≤ Jhat ahat) :
    ∀ a, (1 - εb) * (1 - εn) * J a ≤ (1 + εb) * (1 + εn) * J ahat := by
  intro a
  have hb1 := abs_le.mp (hbias a)
  have hb2 := abs_le.mp (hbias ahat)
  have hn1 := abs_le.mp (hnoise a)
  have hn2 := abs_le.mp (hnoise ahat)
  have hm := hmax a
  have t1 : (1 - εb) * J a ≤ Jdelta a := by linarith [hb1.1]
  have t2 : Jdelta ahat ≤ (1 + εb) * J ahat := by linarith [hb2.2]
  have s1 : (1 - εn) * ((1 - εb) * J a) ≤ (1 - εn) * Jdelta a :=
    mul_le_mul_of_nonneg_left t1 (by linarith)
  have s2 : (1 + εn) * Jdelta ahat ≤ (1 + εn) * ((1 + εb) * J ahat) :=
    mul_le_mul_of_nonneg_left t2 (by linarith)
  have s3 : (1 - εn) * Jdelta a ≤ Jhat a := by linarith [hn1.1]
  have s4 : Jhat ahat ≤ (1 + εn) * Jdelta ahat := by linarith [hn2.2]
  nlinarith [s1, s2, s3, s4, hm]

theorem selector_oracle_bias_noise_witness :
    ∃ (J Jdelta Jhat : Bool → ℝ) (εb εn : ℝ) (ahat : Bool),
      0 ≤ εb ∧ 0 ≤ εn ∧ εn ≤ 1 ∧ (∀ a, 0 < J a) ∧ (∀ a, |Jdelta a - J a| ≤ εb * J a) ∧
        (∀ a, |Jhat a - Jdelta a| ≤ εn * Jdelta a) ∧ (∀ a, Jhat a ≤ Jhat ahat) := by
  refine ⟨fun a => if a then 2 else 1, fun a => if a then 1.8 else 0.95,
    fun a => if a then 1.8 else 0.95, 1 / 10, 1 / 10, true,
    by norm_num, by norm_num, by norm_num, ?_, ?_, ?_, ?_⟩
  · intro a; cases a <;> norm_num
  · intro a; cases a <;> norm_num [abs_le]
  · intro a; cases a <;> norm_num
  · intro a; cases a <;> norm_num

/-- Additive form: a criterion within `η` of `J` selects a basis within `2 η` of the best. -/
theorem selector_additive {ι : Type*} (J Jhat : ι → ℝ) (η : ℝ) (ahat : ι)
    (hclose : ∀ a, |Jhat a - J a| ≤ η)
    (hmax : ∀ a, Jhat a ≤ Jhat ahat) :
    ∀ a, J a - 2 * η ≤ J ahat := by
  intro a
  have h1 := abs_le.mp (hclose a)
  have h2 := abs_le.mp (hclose ahat)
  have hm := hmax a
  linarith [h1.1, h2.2]

theorem selector_additive_witness :
    ∃ (J Jhat : Bool → ℝ) (η : ℝ) (ahat : Bool),
      (∀ a, |Jhat a - J a| ≤ η) ∧ (∀ a, Jhat a ≤ Jhat ahat) := by
  refine ⟨fun a => if a then 2 else 1, fun a => if a then 2.1 else 0.95, 1 / 10, true, ?_, ?_⟩
  · intro a; cases a <;> norm_num [abs_le]
  · intro a; cases a <;> norm_num

/-- Selection consistency: if the criterion is within `η` of its limit `Jlim` and the limit has
a maximiser `alim` separated from the rest by more than `2 η`, the selector returns `alim`. -/
theorem selector_consistent {ι : Type*} (Jlim Jhat : ι → ℝ) (η : ℝ) (alim ahat : ι)
    (hclose : ∀ a, |Jhat a - Jlim a| ≤ η)
    (hgap : ∀ a, a ≠ alim → Jlim a + 2 * η < Jlim alim)
    (hmax : ∀ a, Jhat a ≤ Jhat ahat) :
    ahat = alim := by
  by_contra hne
  have hg := hgap ahat hne
  have h1 := abs_le.mp (hclose alim)
  have h2 := abs_le.mp (hclose ahat)
  have hm := hmax alim
  linarith [h1.1, h2.2]

theorem selector_consistent_witness :
    ∃ (Jlim Jhat : Bool → ℝ) (η : ℝ) (alim ahat : Bool),
      (∀ a, |Jhat a - Jlim a| ≤ η) ∧ (∀ a, a ≠ alim → Jlim a + 2 * η < Jlim alim) ∧
        (∀ a, Jhat a ≤ Jhat ahat) := by
  refine ⟨fun a => if a then 1 else 0, fun a => if a then 1 else 0, 1 / 4, true, true, ?_, ?_, ?_⟩
  · intro a; cases a <;> norm_num
  · intro a ha
    cases a
    · norm_num
    · exact absurd rfl ha
  · intro a; cases a <;> norm_num

/-- The mechanism of a persistent lag: when the limit of the criterion is maximised at `alim`
but the target variance `V` is strictly smaller at `abest`, the selected basis is strictly worse
than `abest`, for every noise level below half the gap of the limit criterion. -/
theorem selector_regret_of_biased_limit {ι : Type*} (V Jlim Jhat : ι → ℝ) (η : ℝ)
    (alim ahat abest : ι)
    (hclose : ∀ a, |Jhat a - Jlim a| ≤ η)
    (hgap : ∀ a, a ≠ alim → Jlim a + 2 * η < Jlim alim)
    (hmax : ∀ a, Jhat a ≤ Jhat ahat)
    (hlag : V abest < V alim) :
    V abest < V ahat := by
  have h := selector_consistent Jlim Jhat η alim ahat hclose hgap hmax
  rw [h]
  exact hlag

theorem selector_regret_of_biased_limit_witness :
    ∃ (V Jlim Jhat : Bool → ℝ) (η : ℝ) (alim ahat abest : Bool),
      (∀ a, |Jhat a - Jlim a| ≤ η) ∧ (∀ a, a ≠ alim → Jlim a + 2 * η < Jlim alim) ∧
        (∀ a, Jhat a ≤ Jhat ahat) ∧ V abest < V alim := by
  refine ⟨fun a => if a then 2 else 1, fun a => if a then 1 else 0, fun a => if a then 1 else 0,
    1 / 4, true, true, false, ?_, ?_, ?_, by norm_num⟩
  · intro a; cases a <;> norm_num
  · intro a ha
    cases a
    · norm_num
    · exact absurd rfl ha
  · intro a; cases a <;> norm_num

/-- One Newton step from a pilot `βp` with the linearised estimating function
`z - c (βp - β) / s` and an assumed slope `c'`: the pilot error survives with the share
`1 - c / c'`, and cancels exactly when `c' = c`. -/
theorem one_step_pilot_share (βp β s z c c' : ℝ) (hs : s ≠ 0) (hc' : c' ≠ 0) :
    βp + s * (z - c * (βp - β) / s) / c' - β = (1 - c / c') * (βp - β) + s * z / c' := by
  field_simp
  ring

theorem one_step_pilot_share_witness :
    ∃ (s c' : ℝ), s ≠ 0 ∧ c' ≠ 0 := by
  exact ⟨1, 1, one_ne_zero, one_ne_zero⟩

/-- Cauchy–Schwarz in the inner product of a positive definite `F`: weights `h` that are not
proportional to `F⁻¹b` give `h'Fh / (h'b)² ≥ 1 / (b'F⁻¹b)`. -/
theorem misweighted_variance_bound {S : ℕ} (F : Matrix (Fin S) (Fin S) ℝ) (hF : F.PosDef)
    (h b : Fin S → ℝ) :
    (h ⬝ᵥ b) ^ 2 ≤ (h ⬝ᵥ F *ᵥ h) * (b ⬝ᵥ F⁻¹ *ᵥ b) := by
  have hT : Fᵀ = F := hF.isHermitian
  have hdet : IsUnit F.det := isUnit_iff_ne_zero.2 hF.det_pos.ne'
  have hFc : F *ᵥ (F⁻¹ *ᵥ b) = b := by
    rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv F hdet, Matrix.one_mulVec]
  generalize hc : F⁻¹ *ᵥ b = c at hFc ⊢
  have hcF : c ᵥ* F = b := by
    conv_lhs => rw [← hT]
    rw [Matrix.vecMul_transpose, hFc]
  have hq : ∀ s : ℝ, 0 ≤ (b ⬝ᵥ c) * (s * s) + (-2 * (h ⬝ᵥ b)) * s + (h ⬝ᵥ F *ᵥ h) := by
    intro s
    have h0 := hF.posSemidef.dotProduct_mulVec_nonneg (h - s • c)
    have e1 : c ⬝ᵥ F *ᵥ h = h ⬝ᵥ b := by
      rw [Matrix.dotProduct_mulVec, hcF, dotProduct_comm]
    have e2 : (h - s • c) ⬝ᵥ F *ᵥ (h - s • c)
        = h ⬝ᵥ F *ᵥ h - 2 * s * (h ⬝ᵥ b) + s * s * (b ⬝ᵥ c) := by
      rw [Matrix.mulVec_sub, Matrix.mulVec_smul, hFc, sub_dotProduct, dotProduct_sub,
        dotProduct_sub, smul_dotProduct, smul_dotProduct, dotProduct_smul, dotProduct_smul, e1,
        dotProduct_comm c b]
      simp only [smul_eq_mul]
      ring
    rw [star_trivial, e2] at h0
    nlinarith [h0]
  have := discrim_le_zero hq
  unfold discrim at this
  nlinarith [this]

theorem misweighted_variance_bound_witness :
    ∃ F : Matrix (Fin 1) (Fin 1) ℝ, F.PosDef := by
  exact ⟨1, Matrix.PosDef.one⟩

end TauPATP

#print axioms TauPATP.selector_oracle
#print axioms TauPATP.selector_oracle_witness
#print axioms TauPATP.selector_variance_regret
#print axioms TauPATP.selector_variance_regret_witness
#print axioms TauPATP.selector_oracle_bias_noise
#print axioms TauPATP.selector_oracle_bias_noise_witness
#print axioms TauPATP.selector_additive
#print axioms TauPATP.selector_additive_witness
#print axioms TauPATP.selector_consistent
#print axioms TauPATP.selector_consistent_witness
#print axioms TauPATP.selector_regret_of_biased_limit
#print axioms TauPATP.selector_regret_of_biased_limit_witness
#print axioms TauPATP.one_step_pilot_share
#print axioms TauPATP.one_step_pilot_share_witness
#print axioms TauPATP.misweighted_variance_bound
#print axioms TauPATP.misweighted_variance_bound_witness
