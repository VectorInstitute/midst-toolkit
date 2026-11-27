# TabDiff Berka + diabetes review rubric

Use with the independent review prompt. Score every seeded item **and** every gap you discover.

## Scale (all dimensions)

| Score | Meaning |
|------:|---------|
| 5 | Correct, evidenced, tested, low residual risk |
| 4 | Correct with minor gaps (docs/tests thin, risk low) |
| 3 | Plausible but weakly evidenced or only golden-covered |
| 2 | Partial / inconsistent; real failure mode plausible |
| 1 | Wrong or unsafe for stated goals |
| 0 | Not checked |

**Pass bar (recommended):** no seeded decision ≤2; no discovered blocker; mean coverage ≥3.5; at least 5 novel gaps listed (even if scored N/A severity).

---

## A. Decision rubrics

Score each decision on **Soundness**, **Fit-to-goal**, **Evidence**, **Residual risk** (each 0–5).
**Decision total** = average of the four (report all four + average).

### A1. Encoded CSV is test “raw” (no CDTD clinical path)

| Dimension | 5 | 3 | 1 |
|-----------|---|---|---|
| Soundness | Vendored `process_data` never requires clinical/`raw_data_path` for diabetes fixtures | Gated/special-cased; easy to regress | Clinical preprocess still on hot path |
| Fit-to-goal | Matches “competition coded data” testing story | Mixed legacy + competition semantics | Still models UCI/CDTD download flow as primary |
| Evidence | Code + tests show `diabetes` works with only `data_path`/`test_path` | Code only, no test | Assumption only in worklog |
| Residual risk | Reference CDTD clearly isolated; no name collision hazard | Same name `"diabetes"` could confuse | Easy to call wrong path in CI/scripts |

### A2. Berka-style coded LabelEncoders (transform-only, column keys)

| Dimension | 5 | 3 | 1 |
|-----------|---|---|---|
| Soundness | Input dtype matches `classes_`; no dataset-specific remaps | Works for Berka; diabetes type/key assumptions implicit | String pickle used on int CSV (or reverse) |
| Fit-to-goal | Enables shared TabDiff API for Berka + diabetes | API shared but contracts differ per fixture | Per-dataset adapter logic crept back |
| Evidence | Unit tests for load/missing keys + integration both fixtures | Integration only | Manual smoke only |
| Residual risk | Unseen codes / empty cats handled or explicitly out of scope | Unknown → sklearn error untested | Silent mis-encode possible |

### A3. Mother (50 w/ IDs) vs fixture (48, no IDs, 128/40)

| Dimension | 5 | 3 | 1 |
|-----------|---|---|---|
| Soundness | Clear provenance; fixture is a documented slice of encoded mother | Slice exists; provenance informal | Fixture schema ≠ mother feature schema |
| Fit-to-goal | IDs excluded for TabDiff; sizes fit golden protocol | Sizes ok; schema extras unexplained | Too large/small or IDs leak into model inputs |
| Evidence | Scripts/paths + column lists match DI `trans` features | Columns match by inspection once | No check against DI/tabddpm_1 |
| Residual risk | Sampling bias / missing rare levels documented | Rare levels absent, unstated | Fixture not representative in a way that hides bugs |

### A4. Fixture-local diabetes vocab vs full-mother vocab

| Dimension | 5 | 3 | 1 |
|-----------|---|---|---|
| Soundness | Train/test codes ⊆ encoder classes; intentional local vocab | Local vocab; membership of codes unchecked in tests | Test can see codes absent from pickle |
| Fit-to-goal | Local vocab justified for model size / golden speed | Justified in prose only | Full-mother vocab makes tests unusable or flaky |
| Evidence | Assert round-trip on fixture; document “not production vocab” | Implied by script | No documentation of which pickle is which |
| Residual risk | Production misuse of fixture pickle unlikely | Two pickles exist; easy to swap | Fixture pickle shipped as competition artifact |

### A5. Diabetes task = regression on `time_in_hospital_integer`

| Dimension | 5 | 3 | 1 |
|-----------|---|---|---|
| Soundness | Matches chosen external protocol (e.g. DI whitebox) and Info indices | Indices consistent internally only | Target index/name mismatch with CSV |
| Fit-to-goal | Serves TabDiff characterization goal stated by team | Convenient but not aligned with attack/`readmitted` goals | Wrong task for MIDST use case |
| Evidence | Compared to DI `data_types`/`meta_data` or written decision | Worklog mention only | No rationale |
| Residual risk | y_policy/regression branches tested | Regression golden passes; unit thin on y | Target treated as cat somewhere |

### A6. Age treated as categorical in DI test schema

| Dimension | 5 | 3 | 1 |
|-----------|---|---|---|
| Soundness | Consistent across CSV, Info, encoders | Consistent in fixture; conflicts with v2 notes | Age in both num and cat, or missing encoder |
| Fit-to-goal | Intentionally DI-faithful | Unresolved product decision | Accidental |
| Evidence | Explicit note in DESIGN/worklog + matching assets | Assets only | Contradictory comments in code |
| Residual risk | Low confusion for future agents | Agents may “fix” age to num and break goldens | Schema drift likely |

### A7. `require_supplied_splits` protects membership

| Dimension | 5 | 3 | 1 |
|-----------|---|---|---|
| Soundness | Missing/empty test_path fails; no silent resplit when flag set | Flag honored in main path; holes in dcr/adult branches | Resplit still possible with flag |
| Fit-to-goal | Membership boundary explicit for both fixtures | Berka only tested | Flag unused by diabetes |
| Evidence | Unit tests for failure + happy path both datasets | One dataset / one failure mode | No test |
| Residual risk | All entrypoints respect flag | Only `process_data` checked | Other loaders bypass |

### A8. Dual golden protocol coherence (seed 0, CPU, 2 ep, 20 samp, deterministic)

| Dimension | 5 | 3 | 1 |
|-----------|---|---|---|
| Soundness | Same knobs both fixtures; determinism test both | Same knobs; determinism only Berka historically | Protocol differs silently per fixture |
| Fit-to-goal | Runtime acceptable; signal useful for both | Diabetes heavier but ok | Diabetes goldens too weak/strong vs Berka |
| Evidence | Parametrized tests + documented protocol | Tests pass; protocol only in old tomorrow docs | No shared protocol doc |
| Residual risk | Platform drift understood (CI vs local) | Known CI strictness asymmetry | Flaky undiagnosed |

---

## B. Coverage rubrics

For each coverage row score **Exists**, **Strength**, **Signal** (0–5).
**Coverage total** = average of three.

| Score | Exists | Strength | Signal |
|------:|--------|----------|--------|
| 5 | Automated test in unit or integration | Asserts precise contract / failure mode | Would catch a real regression from this work |
| 3 | Indirect via goldens or one happy path | Weak asserts (smoke / shape only) | Might catch gross breakages |
| 1 | Mentioned in docs/plans only | No asserts | Decorative |
| 0 | Absent | — | — |

### Seeded coverage rows (score each)

| ID | Area |
|----|------|
| C1 | Prefitted encoder load: missing keys / bad type / missing file |
| C2 | Prefitted transform dtype / coded-int identity (Berka-style) |
| C3 | Encoder path resolution (relative Info path → absolute) |
| C4 | Train-only quantile / no fit on concat (num) |
| C5 | Cat path does not refit when pickle supplied |
| C6 | `require_supplied_splits` missing test_path errors |
| C7 | Berka end-to-end golden preprocess |
| C8 | Berka golden train+sample + determinism |
| C9 | Diabetes end-to-end golden preprocess |
| C10 | Diabetes golden train+sample + determinism |
| C11 | Unit tests load diabetes fixture (not Berka-only conftest) |
| C12 | `process_data("diabetes")` without `raw_data_path` |
| C13 | Inverse / sample column schema matches fixture (both datasets) |
| C14 | Num NaN policy / “no NaNs skipping” both fixtures |
| C15 | Golden `normalize_info` includes encoder path stability |

### Discovered coverage rows

For **each** novel gap you find, add `C-D1…` with the same Exists/Strength/Signal rubric, plus one-line **why it matters**.

Minimum: **5 discovered rows**.

---

## C. Blind-spot / novel-gap severity

When listing “Coverage not considered in the prompt”:

| Severity | Criteria |
|----------|----------|
| Blocker | Can corrupt membership, silently wrong codes, or false-green goldens |
| High | Untested branch on default preprocess path for a fixture you ship |
| Medium | Edge case or second dataset asymmetry |
| Low | Docs/naming/quality |
| Info | Out of scope but noted |

---

## D. Findings severity

| Severity | Criteria |
|----------|----------|
| Blocker | Must fix before trusting goldens / MIDST use |
| High | Fix before merge if this work is the characterization basis |
| Medium | Should fix soon; workaround exists |
| Low | Nit / clarity |
| Note | Observation, not a defect |

Each finding: **severity**, **path**, **repro or reason**, **fix sketch**, **linked rubric IDs** (e.g. A2, C5).

---

## E. Rollup

Report:

1. Mean decision score (A1–A8 averages’ mean)
2. Mean seeded coverage score (C1–C15)
3. Count discovered gaps by severity
4. **Ship rule:**
   - **Blocked** if any decision Soundness ≤2 or any Blocker finding
   - **Ship-with-fixes** if any High finding or mean coverage <3.5
   - **Ship** otherwise (still list Medium/Low)

---

## F. Review completeness gate (for the reviewing agent)

Mark review **incomplete** unless all are true:

- [ ] All A1–A8 scored with four dimensions
- [ ] All C1–C15 scored
- [ ] ≥5 discovered coverage rows (C-D*)
- [ ] Blind-spot section present
- [ ] Full data-path map Berka vs diabetes attached or summarized
- [ ] Explicit “what I did not check”
