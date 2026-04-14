# Annotation Guidelines — Ancient Medical Corpus v1.0

## Overview

This document describes the decision rules applied for each of the eight annotation variables in the corpus. All annotations were performed by the author on English translations of primary sources, with reference to the original language editions for ambiguous passages.

---

## General inclusion criteria

A case was included if it satisfied **all three** of the following:

1. The text describes an individual patient or a well-defined clinical scenario during a febrile illness.
2. Sweating during or immediately around the disease crisis is explicitly mentioned.
3. An outcome (recovery or death) is either stated or clearly implied by the narrative context.

Cases meeting (1) and (2) but not (3) were excluded.

---

## Layer 1 — Primary annotation variables

### H1 — Anatomical location of sweat

**Definition:** The primary anatomical zone from which sweating is described as occurring.

| Value | Label | Annotation rule |
|-------|-------|-----------------|
| 0 | head | Sweating described as occurring on the head, face, forehead, neck, or collar bones primarily |
| 1 | thorax | Sweating described as occurring on the chest, upper body, or back primarily |
| 2 | extremities | Sweating described as occurring on the hands, feet, ankles, or limbs primarily |
| 3 | general | Sweating described as covering the whole body, all over, or without anatomical restriction |

**Decision rule for ambiguous cases:** When multiple zones are described, the label reflects the dominant zone. If distribution is truly equal, label as `general` (3).

---

### H2 — Sweat temperature

**Definition:** The thermal quality of the sweat as described in the text.

| Value | Label | Annotation rule |
|-------|-------|-----------------|
| 0 | cold | Text uses terms: cold, clammy, cannot warm, frigid, icy, or equivalent |
| 1 | tepid | Text uses terms: moderate, lukewarm, neither warm nor cold, or equivalent |
| 2 | warm | Text uses terms: warm, hot, abundant, copious (when warmth is implied), or equivalent |

**Default rule:** When temperature is not explicit but the sweat is described as profuse and crisis-resolving, annotate as `warm` (2). When temperature is not explicit but the crisis fails, annotate as `cold` (0). When neither applies, annotate as `tepid` (1).

---

### H3 — Critical-day timing

**Definition:** Whether the sweating crisis occurs on a day recognized as a critical day in the tradition of origin.

| Value | Label | Annotation rule |
|-------|-------|-----------------|
| 0 | critical day | Crisis occurs on day 7, 11, 14, 20, 21, 40 (Greek/Galenic); equivalent patterned days in other traditions |
| 1 | non-critical | Crisis occurs on a day not designated as critical, or day is unspecified |

**Cross-tradition note:** For Babylonian texts, the notion of critical day is approximated by explicit reference to numbered days or lunar calendar markers. For Chinese texts, the six-stage pattern is used as a proxy. For Islamic texts, Avicenna's explicit list of critical days is applied.

---

### H4 — Clinical prognosis ★ (primary outcome variable)

**Definition:** The patient's ultimate clinical outcome as stated or clearly implied.

| Value | Label | Annotation rule |
|-------|-------|-----------------|
| 0 | favorable | Text states recovery, resolution of fever, return of appetite, patient asked for food, fever left completely, or equivalent |
| 1 | fatal | Text states death, died, will die, expired, or describes irreversible deterioration without recovery |

**For normative/prognostic texts** (e.g. Hippocratic Prognostic, Avicenna Canon): when the text presents a conditional prognosis ("if X, then death"), the label is assigned to the condition described in the passage. For passages describing favorable conditions, label = 0; for passages describing fatal conditions, label = 1.

**⚠ Known ambiguity — Gr_Prog_C1:** This case (Hippocratic Prognostic, Chapter 6) presents both favorable and fatal sweating types in a single conditional sentence. Annotated as `favorable` (0) for the warm-sweat branch (grammatically dominant). This case was misclassified in every experimental condition across all hypotheses and is flagged for philological review.

---

### H5 — Tradition of origin

**Definition:** The medical tradition from which the case is drawn. Ground truth for H5 experiments.

| Value | Label |
|-------|-------|
| 0 | Greek (Hippocratic) |
| 1 | Babylonian |
| 2 | Galenic |
| 3 | Islamic (Avicennan) |
| 4 | Chinese |

---

## Layer 2 — Extended annotation variables

### H6 — Extremity temperature (h11_extremid) ★

**Definition:** The thermal state of the extremities (hands, feet, limbs) at or during the crisis.

| Value | Label | Annotation rule |
|-------|-------|-----------------|
| 0 | warm | Extremities described as warm, hot, or no mention of coldness in a case with warm general sweat |
| 1 | cold | Text uses terms: cold extremities, cannot be warmed, livid, reverting extremities (Chinese: jué), hands/feet cold |

**Coverage:** Approximately 64% of cases have explicit lexical basis for H6. The remaining 36% were annotated by implication:
- Cases with warm general sweat and favorable prognosis → `warm` (0)
- Cases with cold partial sweat and fatal prognosis → `cold` (1)
- Cases where the evidence is insufficient → annotated by context majority

**⚠ Note for reviewers:** The 36% implicit annotation introduces uncertainty in H6. This is disclosed in the manuscript and is a priority for future philological review.

---

### H7 — Delirium during crisis (h10_delirio)

**Definition:** Whether the text explicitly describes mental confusion, delirium, or altered consciousness during the febrile crisis.

| Value | Label | Annotation rule |
|-------|-------|-----------------|
| 0 | absent | No mention of delirium, confusion, or mental alteration |
| 1 | present | Text uses terms: delirium, confused, raving, speechless, loss of consciousness, or equivalent |

**Positive rate:** ~6% of 160 cases (10 cases). Severe class imbalance — results for H7 are reported but not considered scientifically informative.

---

### H8 — Fever pattern (h9_fiebre)

**Definition:** The pattern of fever as described in the text.

| Value | Label | Annotation rule |
|-------|-------|-----------------|
| 0 | continuous | Fever described as continuous, unremitting, ardent, or without daily variation |
| 1 | intermittent | Fever described as intermittent, quotidian, tertian, quartan, or occurring on alternating days |
| 2 | remittent | Fever described as remittent, with daily variation but never fully absent |

**Corpus note:** ~92% of cases are annotated as `continuous` (0), because the corpus was built from crisis case histories rather than long-term fever records. This makes H8 a near-trivial classification problem for majority-class predictors.

---

## Inter-annotator note

All 160 cases were annotated by a single annotator (the author). A formal inter-annotator agreement study was not conducted. These guidelines are published to enable independent re-annotation. Researchers wishing to contribute a second annotation layer are encouraged to contact the author.

---

## Sources used for annotation

| Tradition | Primary edition used |
|-----------|---------------------|
| Greek | Jones, W.H.S. (1923). Hippocrates, Vols. I–IV. Loeb Classical Library. |
| Babylonian | Geller, M.J. (2010). Ancient Babylonian Medicine. Wiley-Blackwell. |
| Galenic | Johnston, I. (2011). Galen: On diseases and symptoms. Cambridge University Press. |
| Islamic | Gruner, O.C. (1930). A treatise on the Canon of Medicine of Avicenna. Luzac & Co. |
| Chinese | Luo, X. (1993). Shang Han Lun. New World Press. |

---

*Version 1.0 — April 2026*
