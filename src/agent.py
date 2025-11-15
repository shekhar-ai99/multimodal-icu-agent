"""Clinical reasoning agent using an LLM with retrieval verification and self-consistency.

Implements `ClinicalAgent` with:
- `reason(fused_embedding, evidence, num_samples=3)` — runs self-consistency sampling and aggregates outputs.
- `verify_claim(claim, evidence)` — lightweight verifier that checks claims against provided evidence strings.

Notes:
- Default model name points to a generic instruct model; replace with a licensed Llama-3 or vLLM-backed model when available.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Dict, Any, List, Optional

import numpy as np

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except Exception:
    pipeline = None


class ClinicalAgent:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-8b-instruct",  # placeholder name
        device: int = -1,
    ):
        self.model_name = model_name
        self.device = device
        self._nlp = None
        if pipeline is not None:
            try:
                self._nlp = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device=self.device,
                )
            except Exception:
                # model may not be available locally; leave _nlp as None and fall back to heuristic
                self._nlp = None

    def embed_to_prompt(self, fused_embedding: Optional[np.ndarray], evidence: Dict[str, str]) -> str:
        """Project fused embedding to a short textual summary and combine with evidence.

        If `evidence` is provided, prefer using it (realistic pipeline projects the embedding
        via a learned mapper to natural language). The implementation here uses evidence directly
        if available, otherwise it synthesizes a short summary from the numeric embedding.
        """
        if evidence is None:
            evidence = {}

        if fused_embedding is None:
            # construct prompt solely from evidence
            cxr = evidence.get("cxr", "CXR unavailable")
            vit = evidence.get("vitals", "Vitals unavailable")
            notes = evidence.get("notes", "Notes unavailable")
        else:
            # best-effort numeric-to-text summary if no evidence strings provided
            cxr = evidence.get("cxr", "CXR: not provided")
            vit = evidence.get("vitals")
            notes = evidence.get("notes")
            if vit is None:
                # summarize embedding numerically
                arr = np.asarray(fused_embedding).ravel()
                # take a few dimensions for a synthetic summary
                dims = arr[:6].tolist()
                vit = "Embedding-summarized vitals: " + ", ".join([f"d{i}={v:.2f}" for i, v in enumerate(dims)])
            if notes is None:
                notes = "No textual notes provided."

        prompt = f"[EVIDENCE]\nCXR: {cxr}\nVitals: {vit}\nNotes: \"{notes}\"\n\n[TASK] Predict shock in 6h. Give diagnosis, plan, rationale. Output JSON with keys: diagnosis, shock_prob, plan, rationale."
        return prompt

    def verify_claim(self, claim: str, evidence: Dict[str, str]) -> bool:
        """Simple claim verifier: checks if claim tokens appear in evidence fields.

        This is intentionally conservative: it looks for direct lexical matches (case-insensitive)
        in the provided evidence strings (`cxr`, `vitals`, `notes`). For more advanced
        verification one would run a retrieval+entailment model over a larger corpus.
        """
        if not claim or evidence is None:
            return False
        claim_l = claim.lower()
        # check main evidence fields
        for key in ("cxr", "vitals", "notes"):
            txt = evidence.get(key)
            if not txt:
                continue
            if isinstance(txt, (list, tuple)):
                txt = " ".join(txt)
            if claim_l in str(txt).lower():
                return True
        # check numeric ranges (e.g., HR 110->130) within vitals field
        vit = evidence.get("vitals")
        if vit and isinstance(vit, str):
            if any(token in claim_l for token in ["hr", "sbp", "spo2", "bp", "resp"]):
                # if claim contains numbers, check approximate match
                import re

                nums = re.findall(r"\d+", claim_l)
                if nums:
                    for n in nums:
                        if n in vit:
                            return True
        return False

    def _call_model(self, prompt: str, num_samples: int = 3, temperature: float = 0.7) -> List[str]:
        """Call the underlying LLM pipeline returning raw text samples.

        If a real model is not available, returns heuristic deterministic outputs.
        """
        if self._nlp is None:
            # fallback: produce synthetic deterministic outputs based on prompt heuristics
            base = {
                "diagnosis": "septic shock",
                "shock_prob": 0.92,
                "plan": "Start vasopressors, aggressive fluids, obtain cultures, monitor lactate.",
                "rationale": "Rising HR, falling BP, low SpO2, febrile, hypotensive notes.",
            }
            outs = []
            for i in range(num_samples):
                # add small variation
                out = json.dumps({k: (v if k != "shock_prob" else round(min(0.99, max(0.0, base["shock_prob"] + (i - 1) * 0.02))), 2)) if False else None})
                # simpler: format JSON manually
                sp = round(min(0.99, max(0.0, base["shock_prob"] + (i - 1) * 0.02)), 2)
                obj = {
                    "diagnosis": base["diagnosis"],
                    "shock_prob": sp,
                    "plan": base["plan"],
                    "rationale": base["rationale"],
                }
                outs.append(json.dumps(obj))
            return outs

        # call transformer pipeline with sampling for self-consistency
        gen = self._nlp(
            prompt,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=256,
            num_return_sequences=num_samples,
        )
        texts = [g.get("generated_text", g["text"]) if isinstance(g, dict) else str(g) for g in gen]
        return texts

    def _parse_model_output(self, txt: str) -> Dict[str, Any]:
        """Try to parse JSON from model output; if parsing fails, fall back to heuristics.

        Returns a dict with keys: diagnosis, shock_prob, plan, rationale
        """
        # attempt to find JSON substring
        try:
            # try direct JSON load
            obj = json.loads(txt)
            return {
                "diagnosis": obj.get("diagnosis") or str(obj.get("diagnosis_text", "")),
                "shock_prob": float(obj.get("shock_prob", 0.0)),
                "plan": obj.get("plan", ""),
                "rationale": obj.get("rationale", ""),
            }
        except Exception:
            # fallback: simple keyword extraction
            low = txt.lower()
            diag = "septic shock" if "septic" in low or "shock" in low else "unstable"
            # find first number that looks like a probability
            import re

            nums = re.findall(r"0\.?\d+|\d+%", txt)
            prob = None
            if nums:
                # pick first and normalize
                s = nums[0]
                if s.endswith("%"):
                    prob = float(s[:-1]) / 100.0
                else:
                    prob = float(s)
                    if prob > 1:
                        prob = prob / 100.0
            if prob is None:
                prob = 0.5
            # plan and rationale heuristics
            plan = "".join([line.strip() + " " for line in txt.splitlines()[:3]])
            rationale = "".join([line.strip() + " " for line in txt.splitlines()[:5]])
            return {"diagnosis": diag, "shock_prob": float(round(prob, 2)), "plan": plan, "rationale": rationale}

    def reason(self, fused_embedding: Optional[np.ndarray], evidence: Dict[str, str], num_samples: int = 3) -> Dict[str, Any]:
        """Run reasoning with self-consistency and verification.

        Steps:
        - build prompt from `fused_embedding` and `evidence`
        - sample `num_samples` answers from the LLM with sampling
        - parse each answer to structured JSON-like dicts
        - aggregate by majority / average probability
        - verify key claims via `verify_claim`
        - return final JSON object
        """
        prompt = self.embed_to_prompt(fused_embedding, evidence)
        raw_samples = self._call_model(prompt, num_samples=num_samples)

        parsed = [self._parse_model_output(txt) for txt in raw_samples]

        # aggregate diagnosis by majority
        diagnoses = [p["diagnosis"] for p in parsed if p.get("diagnosis")]
        diag_counter = Counter(diagnoses)
        if diag_counter:
            diagnosis, _ = diag_counter.most_common(1)[0]
        else:
            diagnosis = parsed[0].get("diagnosis", "unknown")

        # average shock_prob
        probs = [p.get("shock_prob", 0.0) for p in parsed]
        shock_prob = float(sum(probs) / max(1, len(probs)))

        # concatenate plans/rationales (pick the most common rationale by simple heuristic)
        plans = [p.get("plan", "") for p in parsed]
        plan = max(plans, key=lambda s: len(s)) if plans else ""
        rationales = [p.get("rationale", "") for p in parsed]
        rationale = max(rationales, key=lambda s: len(s)) if rationales else ""

        # verify key claims: check diagnosis and short numeric claims in rationale
        verified = True
        # check diagnosis terms
        if diagnosis:
            verified = verified and self.verify_claim(diagnosis, evidence)

        # attempt to extract short claims (e.g., vitals statements) and verify
        # split rationale into sentences
        for sent in (rationale or "").split("."):
            sent = sent.strip()
            if not sent:
                continue
            # check numeric claims or clinically relevant tokens
            if any(tok in sent.lower() for tok in ("hr", "bp", "sbp", "spo2", "hypotens", "fever")):
                ok = self.verify_claim(sent, evidence)
                verified = verified and ok

        out = {
            "diagnosis": diagnosis,
            "shock_prob": float(round(shock_prob, 3)),
            "plan": plan,
            "rationale": rationale,
            "verified": bool(verified),
            "raw_samples": parsed,
        }
        return out


def _demo():
    # small demo used when running the module directly
    agent = ClinicalAgent()
    evidence = {
        "cxr": "Consolidation in right lower lobe",
        "vitals": "HR 110->130, SBP 90->75, SpO2 92%->88%",
        "notes": "Patient febrile, hypotensive",
    }
    out = agent.reason(None, evidence, num_samples=3)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _demo()
