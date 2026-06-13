"""Explainability for the TF-IDF + linear model.

Because the classifier is linear over TF-IDF features, every prediction
decomposes EXACTLY into per-word contributions (no LIME/SHAP approximation):

    decision_score(class c) = intercept_c + sum_i  tfidf_i * coef[c, i]

So we can show, globally, which terms define each specialty, and locally, which
words in a given note drove its prediction.

CLI:
    python -m clinical_classifier.explain --model models/tfidf_submission.joblib --global
    python -m clinical_classifier.explain --model models/tfidf_submission.joblib --text "..."
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np


def _parts(model):
    feats = model.named_steps["features"]
    clf = model.named_steps["clf"]
    names = np.asarray(feats.get_feature_names_out())
    return feats, clf, names, list(clf.classes_)


def _word_mask(names):
    """Keep only the human-readable word features (drop char n-grams)."""
    return np.array([n.startswith("word__") for n in names])


def global_top_terms(model, n=15, word_only=True):
    """Top-weighted terms per class (what the model thinks each specialty 'is')."""
    _, clf, names, classes = _parts(model)
    mask = _word_mask(names) if word_only else np.ones(len(names), bool)
    out = {}
    for ci, cls in enumerate(classes):
        coef = clf.coef_[ci].copy()
        coef[~mask] = -np.inf
        top = np.argsort(coef)[::-1][:n]
        out[cls] = [(names[i].replace("word__", ""), float(clf.coef_[ci][i])) for i in top]
    return out


def explain_note(model, text, top_k=12, word_only=True):
    """Per-note explanation: predicted class, full prob distribution, top drivers."""
    feats, clf, names, classes = _parts(model)
    proba = model.predict_proba([text])[0]
    pred_i = int(np.argmax(proba))
    pred = classes[pred_i]

    x = feats.transform([text]).toarray()[0]
    contrib = x * clf.coef_[pred_i]            # exact additive contribution to the winning class
    mask = _word_mask(names) if word_only else np.ones(len(names), bool)
    idx = np.where((contrib != 0) & mask)[0]
    idx = idx[np.argsort(contrib[idx])[::-1][:top_k]]
    drivers = [(names[i].replace("word__", ""), float(contrib[i])) for i in idx]

    return {
        "prediction": pred,
        "proba": {c: float(p) for c, p in sorted(zip(classes, proba), key=lambda z: -z[1])},
        "top_drivers": drivers,
    }


def token_contributions(model, text, target_class=None):
    """Map each unigram word -> its signed contribution to a class.

    target_class: explain toward this class (e.g. the ensemble's pick). Defaults
    to the TF-IDF model's own argmax. Returns (class, proba_dict_sorted_desc,
    {word: contribution}). Bigram/char features still affect the score but aren't
    returned (not word-level).
    """
    feats, clf, names, classes = _parts(model)
    proba = model.predict_proba([text])[0]
    pred_i = list(classes).index(target_class) if target_class is not None else int(np.argmax(proba))
    x = feats.transform([text]).toarray()[0]
    contrib = x * clf.coef_[pred_i]
    word = {}
    for i, n in enumerate(names):
        if n.startswith("word__"):
            term = n[len("word__"):]
            if " " not in term and contrib[i] != 0:   # unigrams only
                word[term] = float(contrib[i])
    proba_sorted = {c: float(p) for c, p in sorted(zip(classes, proba), key=lambda z: -z[1])}
    return classes[pred_i], proba_sorted, word


def _shade(c, maxabs):
    if not c or maxabs == 0:
        return "transparent"
    a = min(abs(c) / maxabs, 1.0) * 0.85
    rgb = "46,160,67" if c > 0 else "248,81,73"   # green supports / red opposes
    return f"rgba({rgb},{a:.2f})"


REVIEW_THRESHOLD = 0.60  # below this top-class confidence -> route to a human


def ensemble_proba(tfidf_proba, transformer_proba, w_tfidf=0.4):
    """Weight-average two class->prob dicts. Returns dict sorted desc, normalized."""
    classes = set(tfidf_proba) | set(transformer_proba)
    blended = {c: w_tfidf * tfidf_proba.get(c, 0.0) + (1 - w_tfidf) * transformer_proba.get(c, 0.0)
               for c in classes}
    s = sum(blended.values()) or 1.0
    return {c: v / s for c, v in sorted(blended.items(), key=lambda kv: -kv[1])}


def highlight_html(model, text, target_class=None):
    """Return (class, proba, contrib, highlighted_paragraph_html)."""
    import html
    import re

    pred, proba, contrib = token_contributions(model, text, target_class=target_class)
    maxabs = max((abs(v) for v in contrib.values()), default=1.0)
    out = []
    for tok in re.split(r"(\s+)", text):
        if tok.strip() == "":
            out.append(tok)
            continue
        c = sum(contrib.get(w, 0.0) for w in re.findall(r"[a-z]{2,}", tok.lower()))
        out.append(f'<span style="background:{_shade(c, maxabs)};border-radius:3px;padding:0 1px" '
                   f'title="{c:+.3f}">{html.escape(tok)}</span>')
    return pred, proba, contrib, "".join(out)


def routing_badge(pred, conf, threshold=REVIEW_THRESHOLD):
    """The deployment decision: auto-route vs flag for human review."""
    if conf >= threshold:
        return (f'<div style="background:#dafbe1;border:1px solid #2ea043;border-radius:6px;'
                f'padding:8px 12px;font-family:system-ui;font-weight:600">'
                f'✅ Auto-route → {pred} <span style="font-weight:400;color:#555">'
                f'({conf:.0%} confidence)</span></div>')
    return (f'<div style="background:#fff8c5;border:1px solid #d4a72c;border-radius:6px;'
            f'padding:8px 12px;font-family:system-ui;font-weight:600">'
            f'⚠️ Flag for human review <span style="font-weight:400;color:#555">'
            f'(top guess {pred}, only {conf:.0%} confidence — below {threshold:.0%})</span></div>')


def _render_block(pred, proba, body, threshold, subtitle=""):
    """Shared HTML: routing decision, confidence bars, highlighted note."""
    conf = max(proba.values())
    bars = "".join(
        f'<div style="margin:2px 0;font-size:13px"><span style="display:inline-block;width:130px">{c}</span>'
        f'<span style="display:inline-block;height:12px;width:{int(p*240)}px;background:#2ea043;'
        f'vertical-align:middle"></span> {p:.0%}</div>' for c, p in proba.items())
    sub = f'<div style="color:#777;font-size:12px;margin-bottom:6px">{subtitle}</div>' if subtitle else ""
    return (f'<div style="font-family:system-ui,sans-serif;max-width:820px;margin:16px 0">'
            f'{sub}{routing_badge(pred, conf, threshold)}'
            f'<div style="color:#555;margin:12px 0">{bars}</div>'
            f'<p style="line-height:2;font-size:15px;background:#fff;padding:10px;'
            f'border:1px solid #eee;border-radius:6px">{body}</p>'
            f'<p style="color:#999;font-size:12px">Green/red = the TF-IDF component\'s word '
            f'evidence for the predicted class. Hover a word for its exact weight.</p></div>')


def render_html(model, text, threshold=REVIEW_THRESHOLD):
    """TF-IDF-only block."""
    pred, proba, _, body = highlight_html(model, text)
    return _render_block(pred, proba, body, threshold)


def render_html_ensemble(model, text, transformer_proba, w_tfidf=0.4, threshold=REVIEW_THRESHOLD):
    """Ensemble block: decision/confidence from the blend, word evidence from TF-IDF."""
    _, tfidf_proba, _ = token_contributions(model, text)
    blended = ensemble_proba(tfidf_proba, transformer_proba, w_tfidf)
    pred = next(iter(blended))                       # argmax (dict is sorted desc)
    _, _, _, body = highlight_html(model, text, target_class=pred)
    sub = (f"Ensemble: {int(w_tfidf*100)}% TF-IDF + {int((1-w_tfidf)*100)}% transformer "
           f"(decision + confidence); word evidence from the TF-IDF component")
    return _render_block(pred, blended, body, threshold, subtitle=sub)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/tfidf_submission.joblib")
    p.add_argument("--global", dest="glob", action="store_true", help="show top terms per class")
    p.add_argument("--text", help="explain a single note")
    p.add_argument("--html", help="write a highlighted HTML page for --text to this path")
    p.add_argument("-n", type=int, default=15)
    args = p.parse_args()
    model = joblib.load(args.model)

    if args.html and args.text:
        with open(args.html, "w") as f:
            f.write(render_html(model, args.text))
        print(f"wrote {args.html}")
        return

    if args.glob:
        print("=== Top terms per specialty (model coefficients) ===")
        for cls, terms in global_top_terms(model, n=args.n).items():
            print(f"\n{cls}:")
            print("  " + ", ".join(t for t, _ in terms))

    if args.text:
        r = explain_note(model, args.text)
        print(f"\nPrediction: {r['prediction']}")
        print("Probabilities: " + ", ".join(f"{c} {p:.2f}" for c, p in r["proba"].items()))
        print("Top words driving this prediction:")
        for term, w in r["top_drivers"]:
            print(f"  {w:+.3f}  {term}")


if __name__ == "__main__":
    main()
