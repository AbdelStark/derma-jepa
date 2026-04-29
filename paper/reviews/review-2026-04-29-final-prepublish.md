# Final Strict Pre-Publication Peer Review

Paper: `paper/main.tex` (preliminary report, v0.1).

Repository state reviewed: `ebda254` on `main`, with sections `00`--`I`,
tables `tab1`--`tab3`, figures `fig1`--`fig3`, and `figures/locked-numbers.json`
re-read in full. Date: 2026-04-29.

Reviewer brief: a final strict academic peer review immediately before the
author publishes the paper as a preliminary arXiv-style report (with the
EXP-009 partition deferred as an explicit follow-up). The author will
publish at this stage with caveats; the review's job is to (i) decide
whether the paper meets that bar, and (ii) flag every defect that strict
academic readers would still call out.

## Verdict

Accept as preliminary report. The contamination caveat is now load-bearing
and surfaced everywhere it needs to surface, the proxy-task design is
rigorously specified, the run archive and verifier are in place, and the
EXP-009 follow-up sketch carries enough operational detail that a third
party could pick it up. Several defects below are minor (writing,
arithmetic harmonisation, decision-table coverage). They do not block
publication of a preliminary report; they are addressed in the
companion polish pass.

## Major Issues

### M1. Sentence fragment in the introduction

`sections/01-introduction.tex` line 13:

> ``The joint-embedding predictive architecture (JEPA) family
> \citep{lecun2022path, assran2023ijepa} explicitly framing prediction in
> representation space as the path to invariance.''

This is not a sentence: ``framing'' is a participle and the clause has no
finite verb. A strict reviewer flags this on first read. Replace
``framing'' with ``frames''.

### M2. EXP-009 decision table has uncovered outcome bands

`sections/08-followup-exp009.tex` (Table~\ref{tab:exp009-decision}) and
`sections/I-exp009-design.tex` partition outcomes as $\geq 0.85$,
$[0.50, 0.80]$, $[0.30, 0.50]$. Two regions are unmapped:

- $(0.80, 0.85)$ -- a near-miss that strict reviewers would expect mapped
  ahead of time, since post-hoc interpretation of a borderline result is
  exactly the failure mode the decision table is meant to prevent.
- Below $0.30$ -- below random; under the directional convention this
  signals \emph{the same below-random inversion observed for OpenAI CLIP
  and DINOv2}, which would be a meaningful negative outcome rather than
  unmapped.

Tighten the table so it covers $[0, 1]$ exhaustively.

### M3. Compute-budget figure inconsistent across two appendices

`sections/F-compute-budget.tex` reports ``approximately $28$ A10G-hours
of compute, plus an additional $\sim$3 A10G-hours absorbed by two
infrastructure-flake re-launches'' for nine primary-tier runs plus eight
seed-sweep runs.
`sections/G-reproducibility-checklist.tex` reports
``Reproduction cost is $\sim$25 A10G-hours for the nine runs plus the
seed sweep''.

Recomputing: $9 \times 100~\text{min} + 8 \times 95~\text{min} \approx 27.7$
hours, so Appendix F is right and Appendix G is off by $\sim$3 hours.
Align Appendix G to Appendix F.

## Minor Issues

### m1. ``Across nine runs'' framing in the abstract

The abstract says ``the only above-random result on the held-out family
across nine runs.'' Strictly, only six of the nine runs evaluate on
\strongholdtwo\ (EXP-001 is matched-mild, EXP-002 is matched-\strongfam,
EXP-003 is held-out-on-\strongholdone). The abstract reads more cleanly
if the count is qualified -- e.g.\ ``the only above-random result on the
held-out third family among the six runs evaluated on it.'' Minor; some
readers will expect this precision.

### m2. JSON-vs-prose rounding for the EXP-004 MSE-reduction figure

`figures/locked-numbers.json` has
`training_loss_dynamics.exp004_dinov2_linear.reduction_pct_quoted = 13`,
while §7.3 quotes ``$12.6\%$''. The verifier's $0.5$\% tolerance keeps the
build green ($12.58$ vs $13$ has delta $0.42$), but the prose value is
the more accurate one. Tighten the JSON to $12.6$ so the locked source of
truth matches what the paper says.

### m3. Numerical-precision inconsistency for DermLIP / BiomedCLIP

The body and abstract carry $0.944 \pm 0.003$ and $0.329 \pm 0.012$;
Tab~3 and Appendix~A carry $0.9435 \pm 0.0029$ and $0.3286 \pm 0.0120$.
Both are derived from the same locked entries and agree under rounding,
but the ``four-decimal in the table, three-decimal in the prose'' split
will draw a question. Acceptable as-is; a reviewer would not block on
this, but flagging it for awareness.

### m4. AUPRC, EER, and FPR-at-fixed-TPR not surfaced in the body

§3.4 commits to reporting AUPRC, EER, and FPR at TPR$=0.8$; the body
defers all three to the run archive. For a preliminary report this is
defensible (the headline is $\auroc$ and the readers' time is finite),
but a single representative line in Table~1 -- e.g.\ DermLIP's AUPRC and
FPR-at-TPR$0.8$ -- would meaningfully strengthen the headline without
costing space. Optional.

### m5. ``DermQuest'' is no longer a live public source

`sections/I-exp009-design.tex` lists ``DermQuest-derived public images''
as a candidate EXP-009 component. DermQuest was retired in 2019 and is
not currently a public archive. The phrasing ``DermQuest-derived'' is
technically defensible (mirrors / archives exist in derivative releases),
but a reader chasing the dataset will hit a dead end. Either drop the
mention or annotate with a pointer to the surviving derivative archive.

### m6. ``Approximate invariance'' framing in the intro

``a property we read, perhaps charitably, as approximate invariance to
many of the photometric and framing nuisances we care about here.'' The
hedge is correct and intellectually honest, but the editorial aside
(``perhaps charitably'') is stylistically informal for an academic
report. Optional softening; some readers will find the candour
refreshing.

## Things The Paper Gets Right

- The contamination caveat now appears in the title, the abstract, the
  introduction's contribution list, the method-adjacent narrative, the
  results figure caption, the analysis section, the limitations list,
  the conclusion, and an entire load-bearing appendix. There is no
  remaining surface where a casual reader could pick up a stronger claim
  by accident.
- The directional-$\auroc$ convention is fixed early
  (\S\ref{sec:method:eval}), explained in the table caption with the
  sign-symmetric raw-cosine fallback, and the inversion finding is
  framed as a deployment-relevant property rather than a free parameter.
- The EXP-009 decision table is committed up front, which prevents
  post-hoc claim re-interpretation -- this is an unusual and valuable
  feature in a preliminary report.
- Verification-by-construction: every quantitative claim traces to
  `figures/locked-numbers.json` and is checked by `verify_claims.py`
  against the run-archive mirror at tolerance $0.001$.
- The ``what is not yet claimed'' subsection (§7.4) is the kind of
  explicit non-claim list that strict reviewers reward.

## Remaining Scientific Limitations (Unchanged From Prior Review)

- The DermLIP positive remains attribution-limited until EXP-009 or an
  equivalent partition is run. The paper now states this so prominently
  that no reviewer can claim the paper overstates.
- Six of nine primary-tier runs are single-seed; the seed sweep covers
  only the two contamination-relevant configurations. Acceptable for a
  preliminary release; venue submission will likely require expanding
  the sweep.
- HAM10000-only evaluation. Acceptable for a preliminary release;
  external validity is honestly bounded.

## Recommendation

Apply the M1--M3 fixes and the m1--m2 fixes (m3--m6 optional), rebuild,
and publish as a preliminary arXiv-style report. The paper does not yet
meet a full peer-reviewed venue bar (TMLR / MIDL / MICCAI / NeurIPS /
ICLR), but it is a credible, falsification-laddered, contamination-aware
preliminary report and a defensible community-feedback solicitation
ahead of the EXP-009 build decision.
