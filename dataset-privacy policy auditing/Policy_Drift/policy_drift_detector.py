#!/usr/bin/env python3
"""
Privacy Policy Drift Detector (PPDD)
=====================================
Detects semantic and regulatory drift between two versions of a privacy
policy from the same organization, with focused analysis on HIPAA- and
COPPA-relevant language. Designed to reduce human manual review effort
in policy auditing pipelines.

Pipeline:
  1. Ingestion       -> load both policy versions (plain text)
  2. Preprocessing   -> normalize, segment into sentences
  3. Alignment       -> match sentences across versions (SequenceMatcher
                        + Jaccard token similarity for fuzzy pairing)
  4. Drift Classification -> ADDED / REMOVED / MODIFIED / UNCHANGED
  5. Regulatory Analysis  -> HIPAA lexicon, COPPA lexicon, obligation-
                             strength and consent-model shift detection
  6. Reporting       -> CSV audit report + console summary

Usage (batch mode, default):
  python3 policy_drift_detector.py
      Scans ./Old and ./New folders. Old policies are named
      "<Company>_Old.pdf" and new policies "<Company>.pdf".
      Pairs them by company name and writes one combined
      drift_report.csv with a company column.

  Options:
      --old-dir  path to Old folder   (default: Old)
      --new-dir  path to New folder   (default: New)
      -o         output CSV path      (default: drift_report.csv)

Dependencies: pip install pdfplumber (falls back to pypdf if absent).

Author: (for Embry-Riddle privacy policy analysis research)
"""

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Regulatory lexicons
# ---------------------------------------------------------------------------
# Each entry: canonical label -> regex pattern (case-insensitive).
# Patterns are grounded in statutory/regulatory terminology:
#   HIPAA: 45 CFR Parts 160/164 (Privacy, Security, Breach Notification Rules)
#   COPPA: 15 U.S.C. 6501-6506; 16 CFR Part 312

HIPAA_LEXICON = {
    "HIPAA (explicit)":            r"\bHIPAA\b|Health Insurance Portability and Accountability Act",
    "Protected Health Information": r"\bprotected health information\b|\bPHI\b",
    "Covered Entity":              r"\bcovered entit(?:y|ies)\b",
    "Business Associate":          r"\bbusiness associate(?:s)?\b|\bBAA\b|business associate agreement",
    "De-identification":           r"\bde-?identif(?:y|ied|ication)\b|\bsafe harbor\b|\bexpert determination\b",
    "Minimum Necessary":           r"\bminimum necessary\b",
    "Breach Notification":         r"\bbreach notification\b|\bnotif(?:y|ication)[^.]{0,60}breach\b",
    "Treatment/Payment/Operations": r"\btreatment, payment,? (?:and|or) (?:health ?care )?operations\b|\bTPO\b",
    "Authorization (HIPAA)":       r"\bauthorization\b[^.]{0,60}\b(?:disclos|use|health)\b",
    "Medical/Health Records":      r"\bmedical record(?:s)?\b|\bhealth record(?:s)?\b|\bhealth data\b|\bhealth information\b",
    "Security Rule Safeguards":    r"\b(?:administrative|physical|technical) safeguard(?:s)?\b|\bencrypt(?:ion|ed)\b[^.]{0,40}\bhealth\b",
    "Right of Access/Amendment":   r"\bright to (?:access|amend|inspect|obtain a copy of)[^.]{0,50}(?:health|medical|PHI)\b",
}

COPPA_LEXICON = {
    "COPPA (explicit)":            r"\bCOPPA\b|Children'?s Online Privacy Protection Act",
    "Children Under 13":           r"\bunder (?:the age of )?13\b|\bage of 13\b|\bchildren under thirteen\b",
    "Verifiable Parental Consent": r"\bverifiable parental consent\b",
    "Parental Consent (general)":  r"\bparent(?:al)?(?:'s)? (?:consent|permission|approval)\b",
    "Parental Rights (review/delete)": r"\bparent(?:s)?\b[^.]{0,80}\b(?:review|delete|refuse|revoke)\b",
    "Child-Directed Service":      r"\bchild-?directed\b|\bdirected (?:to|at) children\b",
    "Persistent Identifiers (children)": r"\bpersistent identifier(?:s)?\b",
    "Actual Knowledge Standard":   r"\bactual knowledge\b[^.]{0,60}\bchild(?:ren)?\b",
    "Data Minimization (children)": r"\bcollect(?:ion)?\b[^.]{0,60}\bchild(?:ren)?\b[^.]{0,60}\bnecessar",
    "Minors/Children (general)":   r"\bchild(?:ren)?\b|\bminor(?:s)?\b|\bkid(?:s)?\b",
}

# Obligation-strength vocabulary: detects weakening/strengthening of legal
# commitments even when regulatory nouns are unchanged.
STRONG_MODALS = {"will", "shall", "must", "required", "require", "requires"}
WEAK_MODALS   = {"may", "might", "can", "could", "reserve", "reserves", "discretion"}
CONSENT_MODELS = {
    "opt-in":  r"\bopt[- ]?in\b|\baffirmative consent\b|\bexpress consent\b|\bprior consent\b",
    "opt-out": r"\bopt[- ]?out\b|\bunless you object\b|\bwithdraw consent\b",
}

SIMILARITY_MODIFIED_THRESHOLD = 0.55   # below this -> treated as remove+add
SIMILARITY_UNCHANGED_THRESHOLD = 0.900

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DriftRecord:
    change_id: int
    change_type: str          # ADDED | REMOVED | MODIFIED
    v1_sentence: str
    v2_sentence: str
    similarity: float
    hipaa_terms_v1: list = field(default_factory=list)
    hipaa_terms_v2: list = field(default_factory=list)
    coppa_terms_v1: list = field(default_factory=list)
    coppa_terms_v2: list = field(default_factory=list)
    obligation_shift: str = ""
    consent_model_shift: str = ""
    regulatory_domain: str = "GENERAL"
    audit_note: str = ""

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

ABBREV = r"(?:U\.S|e\.g|i\.e|etc|Inc|Corp|Ltd|Sec|No|v|Dr|Mr|Ms)\."

def segment_sentences(text: str):
    """Lightweight sentence segmenter tolerant of legal abbreviations."""
    text = re.sub(r"\s+", " ", text.strip())
    protected = re.sub(ABBREV, lambda m: m.group(0).replace(".", "<DOT>"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def normalize(sentence: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", sentence.lower())


def token_set(sentence: str):
    return set(normalize(sentence).split())

# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def hybrid_similarity(a: str, b: str) -> float:
    """Blend of character-level ratio and token Jaccard overlap."""
    seq = SequenceMatcher(None, normalize(a), normalize(b)).ratio()
    ta, tb = token_set(a), token_set(b)
    jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return 0.5 * seq + 0.5 * jac


def align_sentences(v1_sents, v2_sents):
    """
    Greedy best-match alignment. Returns list of (i, j, sim) pairs plus
    unmatched index sets. O(n*m) — adequate for policy-length documents.
    """
    scores = []
    for i, s1 in enumerate(v1_sents):
        for j, s2 in enumerate(v2_sents):
            sim = hybrid_similarity(s1, s2)
            if sim >= SIMILARITY_MODIFIED_THRESHOLD:
                scores.append((sim, i, j))
    scores.sort(reverse=True)

    used_i, used_j, pairs = set(), set(), []
    for sim, i, j in scores:
        if i not in used_i and j not in used_j:
            pairs.append((i, j, sim))
            used_i.add(i)
            used_j.add(j)

    removed = [i for i in range(len(v1_sents)) if i not in used_i]
    added   = [j for j in range(len(v2_sents)) if j not in used_j]
    return pairs, removed, added

# ---------------------------------------------------------------------------
# Regulatory analysis
# ---------------------------------------------------------------------------

def match_lexicon(sentence: str, lexicon: dict):
    return [label for label, pat in lexicon.items()
            if re.search(pat, sentence, re.IGNORECASE)]


def obligation_shift(s1: str, s2: str) -> str:
    t1, t2 = token_set(s1), token_set(s2)
    lost_strong  = (t1 & STRONG_MODALS) - t2
    gained_weak  = (t2 & WEAK_MODALS) - t1
    lost_weak    = (t1 & WEAK_MODALS) - t2
    gained_strong = (t2 & STRONG_MODALS) - t1
    if lost_strong and gained_weak:
        return f"WEAKENED ({'/'.join(sorted(lost_strong))} -> {'/'.join(sorted(gained_weak))})"
    if lost_weak and gained_strong:
        return f"STRENGTHENED ({'/'.join(sorted(lost_weak))} -> {'/'.join(sorted(gained_strong))})"
    return ""


def consent_model(sentence: str) -> str:
    for model, pat in CONSENT_MODELS.items():
        if re.search(pat, sentence, re.IGNORECASE):
            return model
    return ""


def consent_shift(s1: str, s2: str) -> str:
    m1, m2 = consent_model(s1), consent_model(s2)
    if m1 and m2 and m1 != m2:
        return f"{m1} -> {m2}"
    return ""


def domain_of(rec: DriftRecord) -> str:
    h = bool(rec.hipaa_terms_v1 or rec.hipaa_terms_v2)
    c = bool(rec.coppa_terms_v1 or rec.coppa_terms_v2)
    if h and c:
        return "HIPAA+COPPA"
    if h:
        return "HIPAA"
    if c:
        return "COPPA"
    return "GENERAL"


def audit_note(rec: DriftRecord) -> str:
    notes = []
    if rec.change_type == "REMOVED":
        if rec.hipaa_terms_v1:
            notes.append(f"HIPAA language removed: {', '.join(rec.hipaa_terms_v1)}.")
        cs = [t for t in rec.coppa_terms_v1 if t != "Minors/Children (general)"]
        if cs:
            notes.append(f"COPPA language removed: {', '.join(cs)}.")
    elif rec.change_type == "ADDED":
        if rec.hipaa_terms_v2:
            notes.append(f"New HIPAA language introduced: {', '.join(rec.hipaa_terms_v2)}.")
        cs = [t for t in rec.coppa_terms_v2 if t != "Minors/Children (general)"]
        if cs:
            notes.append(f"New COPPA language introduced: {', '.join(cs)}.")
    else:
        lost_h = set(rec.hipaa_terms_v1) - set(rec.hipaa_terms_v2)
        gained_h = set(rec.hipaa_terms_v2) - set(rec.hipaa_terms_v1)
        lost_c = set(rec.coppa_terms_v1) - set(rec.coppa_terms_v2)
        gained_c = set(rec.coppa_terms_v2) - set(rec.coppa_terms_v1)
        if lost_h:
            notes.append(f"HIPAA terms dropped in revision: {', '.join(sorted(lost_h))}.")
        if gained_h:
            notes.append(f"HIPAA terms gained in revision: {', '.join(sorted(gained_h))}.")
        if lost_c:
            notes.append(f"COPPA terms dropped in revision: {', '.join(sorted(lost_c))}.")
        if gained_c:
            notes.append(f"COPPA terms gained in revision: {', '.join(sorted(gained_c))}.")
    if rec.obligation_shift:
        notes.append(f"Obligation strength shift: {rec.obligation_shift}.")
    if rec.consent_model_shift:
        notes.append(f"Consent model changed: {rec.consent_model_shift}.")
    if not notes:
        notes.append("Textual drift with no direct regulatory-lexicon impact.")
    return " ".join(notes)

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def detect_drift(v1_text: str, v2_text: str):
    v1_sents = segment_sentences(v1_text)
    v2_sents = segment_sentences(v2_text)
    pairs, removed_idx, added_idx = align_sentences(v1_sents, v2_sents)

    records, cid = [], 0

    for i, j, sim in sorted(pairs):
        if sim >= SIMILARITY_UNCHANGED_THRESHOLD:
            continue  # unchanged; excluded from audit report
        cid += 1
        rec = DriftRecord(cid, "MODIFIED", v1_sents[i], v2_sents[j], round(sim, 3))
        rec.hipaa_terms_v1 = match_lexicon(rec.v1_sentence, HIPAA_LEXICON)
        rec.hipaa_terms_v2 = match_lexicon(rec.v2_sentence, HIPAA_LEXICON)
        rec.coppa_terms_v1 = match_lexicon(rec.v1_sentence, COPPA_LEXICON)
        rec.coppa_terms_v2 = match_lexicon(rec.v2_sentence, COPPA_LEXICON)
        rec.obligation_shift = obligation_shift(rec.v1_sentence, rec.v2_sentence)
        rec.consent_model_shift = consent_shift(rec.v1_sentence, rec.v2_sentence)
        rec.regulatory_domain = domain_of(rec)
        rec.audit_note = audit_note(rec)
        records.append(rec)

    for i in removed_idx:
        cid += 1
        rec = DriftRecord(cid, "REMOVED", v1_sents[i], "", 0.0)
        rec.hipaa_terms_v1 = match_lexicon(rec.v1_sentence, HIPAA_LEXICON)
        rec.coppa_terms_v1 = match_lexicon(rec.v1_sentence, COPPA_LEXICON)
        rec.regulatory_domain = domain_of(rec)
        rec.audit_note = audit_note(rec)
        records.append(rec)

    for j in added_idx:
        cid += 1
        rec = DriftRecord(cid, "ADDED", "", v2_sents[j], 0.0)
        rec.hipaa_terms_v2 = match_lexicon(rec.v2_sentence, HIPAA_LEXICON)
        rec.coppa_terms_v2 = match_lexicon(rec.v2_sentence, COPPA_LEXICON)
        rec.regulatory_domain = domain_of(rec)
        rec.audit_note = audit_note(rec)
        records.append(rec)

    records.sort(key=lambda r: r.change_id)
    return records, len(v1_sents), len(v2_sents)

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "company", "change_id", "regulatory_domain", "change_type",
    "similarity", "v1_sentence", "v2_sentence",
    "hipaa_terms_v1", "hipaa_terms_v2",
    "coppa_terms_v1", "coppa_terms_v2",
    "obligation_shift", "consent_model_shift", "audit_note",
]

# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(path: str) -> str:
    """Extract text from a PDF using pdfplumber, falling back to pypdf."""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except ImportError:
        sys.exit("ERROR: install a PDF library first: pip install pdfplumber")


def read_policy(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return extract_pdf_text(path)
    with open(path, encoding="utf-8") as f:
        return f.read()

# ---------------------------------------------------------------------------
# Folder pairing:  Old/<Company>_Old.pdf  <->  New/<Company>.pdf
# ---------------------------------------------------------------------------

def company_key(name: str) -> str:
    """Normalize a filename stem to a matching key (case/space tolerant)."""
    stem = os.path.splitext(name)[0]
    stem = re.sub(r"_old$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def pair_folders(old_dir: str, new_dir: str):
    """Return list of (company, old_path, new_path); report unmatched files."""
    def listing(d):
        if not os.path.isdir(d):
            sys.exit(f"ERROR: folder not found: {d}")
        return {company_key(f): os.path.join(d, f)
                for f in sorted(os.listdir(d))
                if f.lower().endswith((".pdf", ".txt"))}

    old_files, new_files = listing(old_dir), listing(new_dir)
    pairs, unmatched = [], []
    for key, new_path in new_files.items():
        if key in old_files:
            company = os.path.splitext(os.path.basename(new_path))[0]
            pairs.append((company, old_files[key], new_path))
        else:
            unmatched.append(new_path)
    unmatched += [p for k, p in old_files.items() if k not in new_files]
    return pairs, unmatched

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for company, r in rows:
            w.writerow([
                company, r.change_id, r.regulatory_domain, r.change_type,
                r.similarity if r.change_type == "MODIFIED" else "",
                r.v1_sentence, r.v2_sentence,
                "; ".join(r.hipaa_terms_v1), "; ".join(r.hipaa_terms_v2),
                "; ".join(r.coppa_terms_v1), "; ".join(r.coppa_terms_v2),
                r.obligation_shift, r.consent_model_shift, r.audit_note,
            ])


def print_company_summary(company, records, n1, n2, elapsed):
    by_dom = {}
    for r in records:
        by_dom[r.regulatory_domain] = by_dom.get(r.regulatory_domain, 0) + 1
    doms = ", ".join(f"{k}={v}" for k, v in sorted(by_dom.items())) or "none"
    print(f"  {company:<24} {elapsed:6.2f}s  sentences {n1}->{n2}  "
          f"drift events: {len(records):<4} ({doms})")


def main():
    ap = argparse.ArgumentParser(
        description="Batch privacy policy drift detector (HIPAA/COPPA focus)")
    ap.add_argument("--old-dir", default="Old", help="Folder with <Company>_Old.pdf files")
    ap.add_argument("--new-dir", default="New", help="Folder with <Company>.pdf files")
    ap.add_argument("-o", "--output", default="drift_report.csv", help="Combined CSV output path")
    args = ap.parse_args()

    pairs, unmatched = pair_folders(args.old_dir, args.new_dir)
    if not pairs:
        sys.exit("ERROR: no matching company pairs found between the two folders.")

    print("=" * 62)
    print(" PRIVACY POLICY DRIFT REPORT (batch)")
    print("=" * 62)

    all_rows, total_events = [], 0
    timings = []                       # (company, seconds)
    run_start = time.perf_counter()

    for company, old_path, new_path in pairs:
        t_start = time.perf_counter()
        t1 = read_policy(old_path)     # Old = version 1 (baseline)
        t2 = read_policy(new_path)     # New = version 2 (revised)
        records, n1, n2 = detect_drift(t1, t2)
        elapsed = time.perf_counter() - t_start

        print_company_summary(company, records, n1, n2, elapsed)
        all_rows.extend((company, r) for r in records)
        total_events += len(records)
        timings.append((company, elapsed))

    total_time = time.perf_counter() - run_start
    write_csv(all_rows, args.output)

    # Per-company processing time as a small companion CSV (paper-friendly)
    timing_path = os.path.splitext(args.output)[0] + "_timing.csv"
    with open(timing_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["company", "seconds"])
        for company, secs in timings:
            w.writerow([company, f"{secs:.3f}"])
        w.writerow(["TOTAL", f"{total_time:.3f}"])

    avg = total_time / len(pairs)
    slowest = max(timings, key=lambda t: t[1])

    print("-" * 62)
    print(f" Companies compared:   {len(pairs)}")
    if unmatched:
        print(f" Unmatched files:      {len(unmatched)} (skipped)")
        for p in unmatched:
            print(f"   - {p}")
    print(f" Total drift events:   {total_events}")
    print(f" Total time:           {total_time:.2f}s "
          f"({avg:.2f}s per company avg; slowest {slowest[0]} @ {slowest[1]:.2f}s)")
    print(f" CSV report:           {args.output}")
    print(f" Timing report:        {timing_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
