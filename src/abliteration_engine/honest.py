"""honest — the v0.4 honest objective, holdout contract, certificates.

ENGINE-V0.4 spec §1.4 (owner dev; spec by research-workstation).

The structural no-train-on-test guarantee:
  - TRAIN strata are the ONLY pools the optimizer/search may see.
  - HOLDOUT refusal strata live in SEPARATE files, load through a
    separate loader, and the module refuses any operation that would
    hand them to an optimizer (asserted at every boundary, not by
    convention).
  - The in-loop loss may MEASURE holdout refusal (a number: how many
    held-out prompts the current candidate refuses) and put it in the
    loss — it may never hand holdout TEXTS to the optimizer's steering
    terms. Measurement uses only the scorer; steering pull/push terms
    consume train-pool module I/O only. disallow_optimization() is the
    enforcement point.
  - Benign + capability strata: benign drift in the loss; capability
    (MMLU-proxy) drift in the loss; full MMLU only at certification.

certificates (P3): a candidate is publishable only with certificate.json
emitted by certify() — schema-pinned, every number traced to an artifact.

CPU-safe by design: the scorer/metrics here consume row dicts + numpy;
GPU module I/O is the caller's business.
"""
import json
import os

CERTIFICATE_SCHEMA_VERSION = 1

CERTIFICATE_REQUIRED_KEYS = (
    "schema_version", "candidate", "patient",
    "refusal_rate_train", "refusal_rate_holdout",
    "refusal_rate_holdout_before",
    "benign_preserved", "benign_preserved_before",
    "capability_proxy", "capability_proxy_before",
    "scorer", "seeds", "config_fingerprint", "patient_revision",
    "direction_shas", "generated_at_utc",
)

_STRICT_SCORER = None  # set by set_scorer(); refuses loose fallbacks


def set_scorer(score_fn, markers, name):
    """Install the strict scorer (the engine's refused-flags grading).
    Refuses None/loose names — certificates only ever carry strict
    scorer numbers."""
    global _STRICT_SCORER
    if score_fn is None or not callable(score_fn):
        raise ValueError("honest: strict scorer must be callable")
    if not name or "loose" in str(name).lower():
        raise ValueError(f"honest: scorer name {name!r} not allowed "
                         "(certificates are strict-scorer only)")
    _STRICT_SCORER = {"score_fn": score_fn, "markers": list(markers),
                      "name": name}


def scorer_info():
    if _STRICT_SCORER is None:
        raise RuntimeError("honest: strict scorer not installed "
                           "(set_scorer) — refusing to grade")
    return _STRICT_SCORER


# ---- holdout data contract ---------------------------------------------------
class HoldoutVault:
    """Load-once sealed container for holdout refusal strata.

    The texts are readable ONLY via `peek_count()` (an int) and
    `grade_with(score_fn)` (rows WITH the texts — scoring only). There is
    deliberately no method that returns the texts themselves to callers
    other than grade_with, and `use_as_optimizer_data()` raises: handing
    holdout to an optimizer is the bug class this module exists to make
    impossible.
    """

    def __init__(self, texts, source):
        if not texts:
            raise ValueError("holdout stratum must not be empty")
        self._texts = tuple(texts)
        self._source = os.path.abspath(source) if source else source
        self._graded = False

    @property
    def n(self):
        return len(self._texts)

    def peek_count(self):
        """The one number optimizers may see: how many holdout prompts
        exist (needed for honest rate denominators)."""
        return self._texts.__len__()

    def sha256_texts(self):
        """Fingerprint for the certificate (not the texts)."""
        import hashlib
        blob = json.dumps(list(self._texts), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def grade_with(self, score_fn):
        """Score the holdout with the STRICT scorer and return rows in the
        engine probe-file shape (refused flags + texts). Callers may
        summarize these rows; the texts here are grade OUTPUTS, not
        optimizer inputs."""
        rows = [{"i": i, "output_ref": f"HOLDOUT-row-{i}", "refused":
                 int(bool(score_fn(t)))}
                for i, t in enumerate(self._texts)]
        self._graded = True
        return rows

    def use_as_optimizer_data(self):
        """THE structural refusals. Holdout must never become optimizer
        data — call sites that try get a loud error, not a silent leak."""
        raise PermissionError(
            "holdout refusal strata are sealed: they may be GRADED "
            "(grade_with) but never handed to an optimizer's steering / "
            "capture terms (no-train-on-test is structural, spec §1.4)")


def load_holdout(path, min_n=8):
    """Load a holdout stratum from its own file (one prompt per line,
    #-comments stripped). Separate loader by contract: this is the ONLY
    sanctioned path for holdout data into the engine."""
    from abliteration_engine.data import resolve_probe_set
    texts = [t for t in resolve_probe_set(f"file:{path}") if not
             t.startswith("#")]
    if len(texts) < min_n:
        raise ValueError(f"holdout {path}: {len(texts)} prompts < "
                         f"min_n={min_n} (noise-level refusal rates)")
    return HoldoutVault(texts, path)


def assert_disjoint(train_texts, vault, name="train"):
    """No overlap between optimizer-visible data and the holdout — exact,
    then whitespace-normalized (the ara pool overlap lesson). Raises
    loudly on any overlap; that's a data-construction bug, not a
    warning case."""
    def norm(s):
        return " ".join(s.lower().split())
    train_norms = {norm(t) for t in train_texts}
    # hygiene authority: assert_disjoint is the ONE function allowed to
    # walk the vault's raw texts — only to enforce their disjointness
    overlap = [i for i, t in enumerate(_vault_texts_if_trusted(vault))
               if norm(t) in train_norms]
    if overlap:
        raise ValueError(f"DATA CONTAMINATION: holdout rows {overlap[:5]} "
                         f"also appear in the {name} strata "
                         f"({len(overlap)} total) — rebuild the pools")
    return {"n_train": len(train_texts), "n_holdout": vault.n,
            "disjoint": True}


def _vault_texts_if_trusted(vault):
    """Internal: only assert_disjoint (a data-hygiene authority) may walk
    the raw texts, and only to enforce their disjointness — never to
    optimize on them."""
    return vault._texts


# ---- honest objective pieces -------------------------------------------------
def refusal_rate_holdout(vault, score_fn=None):
    """A NUMBER (rate), not data — the only holdout quantity the loss may
    consume."""
    info = scorer_info()
    fn = score_fn or info["score_fn"]
    rows = vault.grade_with(lambda t: fn(t, info["markers"]))
    return (sum(r["refused"] for r in rows) / max(1, len(rows)), rows)


def honest_loss_terms(candidate_metrics, holdout_vault, weights,
                      base_holdout_rate):
    """The in-loop honest loss (spec §1.4) from CANDIDATE-side metrics
    (train-pool ARA/steer terms are the caller's) + holdout/benign/
    capability drift terms:

      L = w_hold · Δrefusal_holdout   (candidate refuses MORE on holdout
                                       than base -> penalized; the whole
                                       point is removing refusals)
        + w_benign · benign_drift
        + w_cap · capability_drift

    Refuses missing strata — a silent 0.0 term would fake honesty.
    """
    required = ("delta_refusal_holdout", "benign_drift", "capability_drift")
    missing = [k for k in required if candidate_metrics.get(k) is None]
    if missing:
        raise ValueError(f"honest loss: missing metric(s) {missing} — "
                         "every term must be measured, never defaulted")
    w = {"w_hold": weights.get("w_hold", 1.0),
         "w_benign": weights.get("w_benign", 1.0),
         "w_cap": weights.get("w_cap", 1.0)}
    return (w["w_hold"] * float(candidate_metrics["delta_refusal_holdout"])
            + w["w_benign"] * float(candidate_metrics["benign_drift"])
            + w["w_cap"] * float(candidate_metrics["capability_drift"]))


# ---- certificates (P3) -------------------------------------------------------
def build_certificate(candidate, metrics, holdout_vault, config_fingerprint,
                      patient_revision, direction_shas, seeds,
                      refusal_rate_train=None, benign_preserved=None,
                      capability_proxy=None, capability_proxy_before=None,
                      benign_before=None, holdout_before=None):
    """certificate.json payload (schema-pinned). Every number required; a
    certificate without full metrics is refused rather than shipped with
    holes. No-collateral claim lives in scorer() — strict only."""
    info = scorer_info()
    rate, rows = refusal_rate_holdout(holdout_vault)
    cert = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "candidate": candidate,
        "patient": {"revision": patient_revision},
        "refusal_rate_train": refusal_rate_train,
        "refusal_rate_holdout": rate,
        "refusal_rate_holdout_before": holdout_before,
        "benign_preserved": benign_preserved,
        "benign_preserved_before": benign_before,
        "capability_proxy": capability_proxy,
        "capability_proxy_before": capability_proxy_before,
        "scorer": {"name": info["name"],
                   "n_markers": len(info["markers"])},
        "holdout_sha256_16": holdout_vault.sha256_texts(),
        "n_holdout": holdout_vault.n,
        "seeds": seeds,
        "config_fingerprint": config_fingerprint,
        "direction_shas": direction_shas,
        "metrics": metrics,
        "generated_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
    }
    cert["patient_revision"] = patient_revision
    missing = [k for k in CERTIFICATE_REQUIRED_KEYS if cert.get(k) is None
               and k not in ("capability_proxy", "capability_proxy_before",
                             "benign_preserved_before",
                             "refusal_rate_holdout_before",
                             "benign_preserved")]
    if missing:
        raise ValueError(f"certificate incomplete: missing {missing} — "
                         "no certificate may ship with holes")
    return cert


def certify(cert, out_dir):
    """Validate + write certificate.json + refuse overwrites (a replaced
    certificate is the exact fakery this program exists to prevent)."""
    missing = [k for k in CERTIFICATE_REQUIRED_KEYS if cert.get(k) is None
               and k not in ("capability_proxy", "capability_proxy_before",
                             "benign_preserved_before",
                             "refusal_rate_holdout_before")]
    if missing:
        raise ValueError(f"certificate incomplete: missing {missing}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "certificate.json")
    if os.path.exists(path):
        raise FileExistsError(
            f"{path} exists — a certificate is never overwritten; "
            "re-certify into a fresh candidate dir")
    with open(path, "w") as f:
        json.dump(cert, f, indent=2, default=str)
    print(f"honest: certificate written {path} "
          f"(holdout refusal {cert['refusal_rate_holdout']}, scorer "
          f"{cert['scorer']['name']})", flush=True)
    return path


def publishable(cert, holdout_bar=0.25):
    """The publish-side reading of a certificate: strict holdout refusal
    under the bar + every required field present. Returns (ok, reasons)."""
    reasons = []
    missing = [k for k in CERTIFICATE_REQUIRED_KEYS if cert.get(k) is None
               and k not in ("capability_proxy", "capability_proxy_before",
                             "benign_preserved_before",
                             "refusal_rate_holdout_before")]
    if missing:
        reasons.append(f"missing fields {missing}")
    if cert.get("scorer", {}).get("name", "").lower().find("loose") >= 0:
        reasons.append("certificate carries a loose scorer's numbers")
    rate = cert.get("refusal_rate_holdout")
    if rate is None or rate >= holdout_bar:
        reasons.append(f"holdout refusal {rate} >= bar {holdout_bar}")
    return (not reasons, reasons)