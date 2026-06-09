"""Biomedical data-source clients.

Each function returns a small, JSON-serialisable dict suitable for inclusion
in an LLM prompt. We deliberately trim payloads to keep prompts focused.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Any

from .http import get_json, get_text, post_json
from .mutation import MutationQuery, VariantClass

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MYVARIANT = "https://myvariant.info/v1"
MYGENE = "https://mygene.info/v3"
OPENTARGETS = "https://api.platform.opentargets.org/api/v4/graphql"
CIVIC_GQL = "https://civicdb.org/api/graphql"
MUTALYZER = "https://mutalyzer.nl/api"


# ---------- MyGene.info ----------
def canonicalize_symbol(symbol: str) -> str:
    """Resolve aliases (e.g. HER2 → ERBB2) to the official HGNC symbol.

    Falls back to the upper-cased input on any error or no match.
    """
    if not symbol:
        return symbol
    sym = symbol.strip().upper()
    try:
        # symbol match first (cheap, authoritative)
        res = get_json(
            f"{MYGENE}/query",
            params={"q": f"symbol:{sym}", "species": "human", "size": 1, "fields": "symbol"},
        )
        hits = res.get("hits") or []
        if hits and hits[0].get("symbol"):
            return hits[0]["symbol"]
        # alias fallback
        res = get_json(
            f"{MYGENE}/query",
            params={"q": f"alias:{sym}", "species": "human", "size": 1, "fields": "symbol"},
        )
        hits = res.get("hits") or []
        if hits and hits[0].get("symbol"):
            return hits[0]["symbol"]
    except Exception:  # noqa: BLE001
        pass
    return sym


def gene_info(symbol: str) -> dict[str, Any]:
    """Look up canonical gene metadata."""
    try:
        hit = get_json(
            f"{MYGENE}/query",
            params={
                "q": f"symbol:{symbol}",
                "species": "human",
                "size": 1,
                "fields": "symbol,name,entrezgene,ensembl.gene,summary,alias,type_of_gene",
            },
        )
        hits = hit.get("hits") or []
        if not hits:
            return {"symbol": symbol, "found": False}
        h = hits[0]
        ensembl = h.get("ensembl")
        ensembl_id = ensembl.get("gene") if isinstance(ensembl, dict) else (
            ensembl[0].get("gene") if isinstance(ensembl, list) and ensembl else None
        )
        return {
            "symbol": h.get("symbol"),
            "name": h.get("name"),
            "entrez_id": h.get("entrezgene"),
            "ensembl_gene": ensembl_id,
            "aliases": h.get("alias"),
            "type_of_gene": h.get("type_of_gene"),
            "summary": h.get("summary"),
            "found": True,
        }
    except Exception as e:  # noqa: BLE001
        return {"symbol": symbol, "error": str(e), "found": False}


# ---------- MyVariant.info ----------
def variant_info(mq: MutationQuery) -> dict[str, Any]:
    """Resolve a variant via MyVariant.info using rsid, HGVS, or gene+change."""
    if mq.variant_class in (VariantClass.FUSION, VariantClass.CNV_AMP, VariantClass.CNV_DEL, VariantClass.EXON_SKIP):
        # MyVariant indexes single-nucleotide / small-indel events; these classes
        # have no useful representation there.
        return {"found": False, "reason": f"not applicable for {mq.variant_class}"}

    q: str | None = None
    if mq.rsid:
        q = f"dbsnp.rsid:{mq.rsid}"
    elif mq.hgvs:
        try:
            return _trim_variant(get_json(f"{MYVARIANT}/variant/{mq.hgvs}"))
        except Exception:
            q = mq.hgvs
    elif mq.gene and mq.protein_change:
        q = (
            f'dbnsfp.genename:{mq.gene} AND '
            f'(dbnsfp.aa.altref:{mq.protein_change} OR dbnsfp.hgvsp:p.{mq.protein_change})'
        )
    elif mq.gene:
        q = f"dbnsfp.genename:{mq.gene}"

    if not q:
        return {"found": False, "reason": "no query"}

    try:
        res = get_json(
            f"{MYVARIANT}/query",
            params={
                "q": q,
                "size": 1,
                "fields": "dbsnp,clinvar,cadd,dbnsfp,cosmic,civic,docm,gnomad_exome,gnomad_genome",
            },
        )
        hits = res.get("hits") or []
        if not hits:
            return {"found": False, "query": q}
        return _trim_variant(hits[0])
    except Exception as e:  # noqa: BLE001
        return {"found": False, "error": str(e), "query": q}


def _safe(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _trim_variant(v: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"found": True, "id": v.get("_id")}
    if "dbsnp" in v and isinstance(v["dbsnp"], dict):
        out["rsid"] = v["dbsnp"].get("rsid")
    if "clinvar" in v and isinstance(v["clinvar"], dict):
        cv = v["clinvar"]
        rcv = cv.get("rcv") or []
        if isinstance(rcv, dict):
            rcv = [rcv]
        out["clinvar"] = {
            "variant_id": cv.get("variant_id"),
            "rcv": [_trim_clinvar_rcv(r) for r in rcv[:5]],
        }
    if "cadd" in v and isinstance(v["cadd"], dict):
        out["cadd_phred"] = v["cadd"].get("phred")
        cons = v["cadd"].get("consequence")
        if isinstance(cons, (str, list)):
            out["consequence"] = cons
    if "dbnsfp" in v and isinstance(v["dbnsfp"], dict):
        out["dbnsfp"] = _trim_dbnsfp(v["dbnsfp"])
    if "cosmic" in v and isinstance(v["cosmic"], dict):
        out["cosmic_id"] = v["cosmic"].get("cosmic_id")
    for af_key in ("gnomad_exome", "gnomad_genome"):
        af = _safe(v, af_key, "af", "af")
        if af is not None:
            out[af_key + "_af"] = af
    return out


def _trim_clinvar_rcv(r: dict) -> dict[str, Any]:
    conditions = r.get("conditions")
    if isinstance(conditions, dict):
        conditions = conditions.get("name")
    return {
        "accession": r.get("accession"),
        "clinical_significance": r.get("clinical_significance"),
        "conditions": conditions,
    }


def _trim_dbnsfp(d: dict) -> dict[str, Any]:
    return {
        "gene": d.get("genename"),
        "hgvsp": d.get("hgvsp"),
        "hgvsc": d.get("hgvsc"),
        "sift_pred": _safe(d, "sift", "pred"),
        "polyphen2_hdiv_pred": _safe(d, "polyphen2", "hdiv", "pred"),
        "revel_score": _safe(d, "revel", "score") or d.get("revel"),
        "alphamissense_pred": _safe(d, "alphamissense", "pred"),
        "alphamissense_score": _safe(d, "alphamissense", "score"),
        "clinpred_pred": _safe(d, "clinpred", "pred"),
    }


# ---------- ClinVar via NCBI E-utilities ----------
def clinvar_summary(mq: MutationQuery, max_results: int = 5) -> dict[str, Any]:
    term_parts = []
    if mq.gene:
        term_parts.append(f"{mq.gene}[gene]")
    if mq.protein_change:
        term_parts.append(f'"{mq.protein_change}"')
    if mq.hgvs:
        term_parts.append(f'"{mq.hgvs}"')
    if mq.rsid:
        term_parts.append(mq.rsid)
    if mq.variant_class == VariantClass.EXON_SKIP and mq.exon:
        term_parts.append(f'"exon {mq.exon}"')
    if mq.variant_class == VariantClass.CNV_AMP:
        term_parts.append('("copy number gain" OR amplification)')
    if mq.variant_class == VariantClass.CNV_DEL:
        term_parts.append('("copy number loss" OR deletion)')
    if mq.fusion_partners:
        a, b = mq.fusion_partners
        # Append fusion clauses without discarding any previously appended filters
        # (protein_change / hgvs / rsid). The bare gene[gene] clause is omitted
        # for fusions to avoid duplicating the partner clause below.
        term_parts = [p for p in term_parts if p != f"{mq.gene}[gene]"]
        term_parts.append(f'({a}[gene] OR {b}[gene])')
        term_parts.append("fusion")

    if not term_parts:
        return {"found": False}

    term = " AND ".join(term_parts)
    return _eutils_summary("clinvar", term, max_results, _clinvar_record)


def _clinvar_record(cid: str, doc: dict) -> dict[str, Any]:
    germ = doc.get("germline_classification") or {}
    return {
        "uid": cid,
        "title": doc.get("title"),
        "clinical_significance": germ.get("description")
            or (doc.get("clinical_significance") or {}).get("description"),
        "review_status": germ.get("review_status"),
        "trait_set": [t.get("trait_name") for t in (doc.get("trait_set") or [])][:5],
    }


# ---------- PubMed abstracts ----------
FETCH_PUBMED_ABSTRACTS = os.getenv("FETCH_PUBMED_ABSTRACTS", "1") != "0"
PUBMED_ABSTRACT_CHARS = int(os.getenv("PUBMED_ABSTRACT_CHARS", "1200"))


def pubmed_search(mq: MutationQuery, max_results: int = 6) -> dict[str, Any]:
    parts: list[str] = []
    if mq.fusion_partners:
        a, b = mq.fusion_partners
        parts.append(f'({a}[Gene] AND {b}[Gene])')
        parts.append("fusion")
    elif mq.gene:
        parts.append(f"{mq.gene}[Gene]")

    if mq.protein_change:
        parts.append(f'"{mq.protein_change}"')
    if mq.hgvs:
        parts.append(f'"{mq.hgvs}"')
    if mq.rsid:
        parts.append(mq.rsid)
    if mq.variant_class == VariantClass.EXON_SKIP and mq.exon:
        parts.append(f'(exon {mq.exon} OR "exon {mq.exon} skipping")')
    if mq.variant_class == VariantClass.CNV_AMP:
        parts.append("(amplification OR overexpression)")
    if mq.variant_class == VariantClass.CNV_DEL:
        parts.append("(deletion OR loss)")

    if not parts:
        return {"found": False}
    term = " AND ".join(parts)
    summary = _eutils_summary("pubmed", term, max_results, _pubmed_record, sort="relevance")

    # Enrich with abstract text via efetch. Best-effort; we never fail the
    # whole call if abstract fetching breaks.
    if FETCH_PUBMED_ABSTRACTS and summary.get("found"):
        pmids = [r["pmid"] for r in summary.get("records", []) if r.get("pmid")]
        if pmids:
            abstracts = _pubmed_fetch_abstracts(pmids)
            for r in summary["records"]:
                ab = abstracts.get(r.get("pmid"))
                if ab:
                    r["abstract"] = ab[:PUBMED_ABSTRACT_CHARS]
    return summary


def _pubmed_record(pid: str, doc: dict) -> dict[str, Any]:
    return {
        "pmid": pid,
        "title": doc.get("title"),
        "journal": doc.get("fulljournalname") or doc.get("source"),
        "pubdate": doc.get("pubdate"),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
    }


def _pubmed_fetch_abstracts(pmids: list[str]) -> dict[str, str]:
    """Fetch PubMed abstracts via efetch. Returns {pmid: abstract_text}."""
    if not pmids:
        return {}
    params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if os.getenv("NCBI_EMAIL"):
        params["email"] = os.getenv("NCBI_EMAIL")
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.getenv("NCBI_API_KEY")
    try:
        xml = get_text(f"{NCBI_BASE}/efetch.fcgi", params=params)
        root = ET.fromstring(xml)
        out: dict[str, str] = {}
        for art in root.findall(".//PubmedArticle"):
            pmid_el = art.find(".//PMID")
            if pmid_el is None or not pmid_el.text:
                continue
            pmid = pmid_el.text.strip()
            chunks: list[str] = []
            for ab in art.findall(".//Abstract/AbstractText"):
                label = ab.get("Label")
                # ET.tostring with method="text" captures nested italics/sub etc.
                txt = "".join(ab.itertext()).strip()
                if not txt:
                    continue
                chunks.append(f"{label}: {txt}" if label else txt)
            if chunks:
                out[pmid] = re.sub(r"\s+", " ", " ".join(chunks)).strip()
        return out
    except Exception:  # noqa: BLE001
        return {}


def _eutils_summary(db: str, term: str, max_results: int, record_fn, sort: str | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"db": db, "term": term, "retmode": "json", "retmax": max_results}
    if sort:
        params["sort"] = sort
    if os.getenv("NCBI_EMAIL"):
        params["email"] = os.getenv("NCBI_EMAIL")
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.getenv("NCBI_API_KEY")
    try:
        search = get_json(f"{NCBI_BASE}/esearch.fcgi", params=params)
        ids = (search.get("esearchresult") or {}).get("idlist") or []
        if not ids:
            return {"found": False, "term": term}
        summ = get_json(
            f"{NCBI_BASE}/esummary.fcgi",
            params={"db": db, "id": ",".join(ids), "retmode": "json"},
        )
        result = summ.get("result") or {}
        records = [record_fn(i, result[i]) for i in ids if result.get(i)]
        return {"found": True, "term": term, "records": records}
    except Exception as e:  # noqa: BLE001
        return {"found": False, "error": str(e), "term": term}


# ---------- Open Targets (diseases + drugs by gene) ----------
_OT_QUERY = """
query TargetEvidence($symbol: String!) {
  search(queryString: $symbol, entityNames: ["target"]) {
    hits { id name entity }
  }
}
"""

_OT_TARGET_QUERY = """
query Target($id: String!) {
  target(ensemblId: $id) {
    id
    approvedSymbol
    approvedName
    associatedDiseases(page: {index: 0, size: 8}) {
      rows { disease { id name } score }
    }
    drugAndClinicalCandidates {
      count
      rows {
        id
        maxClinicalStage
        drug {
          id
          name
          drugType
          maximumClinicalStage
          mechanismsOfAction { rows { mechanismOfAction } }
        }
      }
    }
  }
}
"""


def opentargets_for_gene(symbol: str) -> dict[str, Any]:
    try:
        s = post_json(OPENTARGETS, {"query": _OT_QUERY, "variables": {"symbol": symbol}})
        hits = (((s.get("data") or {}).get("search") or {}).get("hits")) or []
        target_hit = next((h for h in hits if h.get("entity") == "target"), None)
        if not target_hit:
            return {"found": False, "symbol": symbol}
        tid = target_hit["id"]
        t = post_json(OPENTARGETS, {"query": _OT_TARGET_QUERY, "variables": {"id": tid}})
        target = ((t.get("data") or {}).get("target")) or {}
        diseases = [
            {"id": r["disease"]["id"], "name": r["disease"]["name"], "score": r.get("score")}
            for r in (_safe(target, "associatedDiseases", "rows") or [])
        ]
        drugs: list[dict[str, Any]] = []
        for r in (_safe(target, "drugAndClinicalCandidates", "rows") or [])[:15]:
            drug = r.get("drug") or {}
            moas = _safe(drug, "mechanismsOfAction", "rows") or []
            drugs.append({
                "drug": drug.get("name"),
                "drug_id": drug.get("id"),
                "drug_type": drug.get("drugType"),
                "mechanism": "; ".join(
                    m.get("mechanismOfAction") for m in moas if m.get("mechanismOfAction")
                ) or None,
                "max_clinical_stage": r.get("maxClinicalStage") or drug.get("maximumClinicalStage"),
            })
        return {
            "found": True,
            "ensembl_id": tid,
            "approved_symbol": target.get("approvedSymbol"),
            "approved_name": target.get("approvedName"),
            "associated_diseases": diseases,
            "known_drugs": drugs,
            "known_drugs_total": _safe(target, "drugAndClinicalCandidates", "count"),
        }
    except Exception as e:  # noqa: BLE001
        return {"found": False, "symbol": symbol, "error": str(e)}


# ---------- CIViC (variant-level therapy evidence) ----------
_CIVIC_QUERY = """
query GeneVariants($symbol: String!) {
  gene(entrezSymbol: $symbol) {
    id
    name
    description
    variants(first: 50) {
      nodes {
        id
        name
        variantAliases
        evidenceItems(first: 25, status: ACCEPTED) {
          nodes {
            id
            evidenceLevel
            evidenceType
            evidenceDirection
            significance
            description
            disease { name doid }
            therapies { name ncitId }
            therapyInteractionType
            source { citationId sourceType }
          }
        }
      }
    }
  }
}
"""

_CIVIC_MAX_EVIDENCE_PER_VARIANT = int(os.getenv("CIVIC_MAX_EVIDENCE_PER_VARIANT", "8"))


def _trim_civic_evidence(ev: dict) -> dict[str, Any]:
    disease = ev.get("disease") or {}
    therapies = ev.get("therapies") or []
    src = ev.get("source") or {}
    return {
        "id": ev.get("id"),
        "evidence_level": ev.get("evidenceLevel"),
        "evidence_type": ev.get("evidenceType"),
        "direction": ev.get("evidenceDirection"),
        "significance": ev.get("significance"),
        "disease": disease.get("name") if isinstance(disease, dict) else None,
        "disease_doid": disease.get("doid") if isinstance(disease, dict) else None,
        "therapies": [t.get("name") for t in therapies if t.get("name")] or None,
        "therapy_interaction": ev.get("therapyInteractionType"),
        "source": f"{src.get('sourceType','?')}:{src.get('citationId')}"
            if src.get("citationId") else None,
        "description": (ev.get("description") or "")[:400] or None,
    }


def civic_for_gene(symbol: str, variant_hint: str | None = None) -> dict[str, Any]:
    """CIViC evidence for a gene (entrez symbol). Best-effort; degrades gracefully."""
    try:
        res = post_json(CIVIC_GQL, {"query": _CIVIC_QUERY, "variables": {"symbol": symbol}})
        gene = _safe(res, "data", "gene")
        if not gene:
            return {"found": False, "symbol": symbol}
        variants = _safe(gene, "variants", "nodes") or []
        records: list[dict[str, Any]] = []
        for v in variants:
            ev_nodes = _safe(v, "evidenceItems", "nodes") or []
            evidence = [_trim_civic_evidence(e) for e in ev_nodes[:_CIVIC_MAX_EVIDENCE_PER_VARIANT]]
            records.append({
                "id": v.get("id"),
                "name": v.get("name"),
                "aliases": v.get("variantAliases"),
                "url": f"https://civicdb.org/variants/{v.get('id')}",
                "evidence_count": len(ev_nodes),
                "evidence": evidence or None,
            })
        if variant_hint:
            hint = variant_hint.upper()
            filtered = [
                r for r in records
                if hint in (r["name"] or "").upper()
                or any(hint in (a or "").upper() for a in (r["aliases"] or []))
            ]
            if filtered:
                records = filtered
        return {
            "found": True,
            "gene_id": gene.get("id"),
            "gene_url": f"https://civicdb.org/genes/{gene.get('id')}",
            "description": (gene.get("description") or "")[:500],
            "variants": records[:10],
        }
    except Exception as e:  # noqa: BLE001
        return {"found": False, "symbol": symbol, "error": str(e)}


# ---------- Optional HGVS normalisation (Mutalyzer 3 REST) ----------
def normalize_hgvs(hgvs: str) -> dict[str, Any]:
    """Normalise / validate a fully-qualified HGVS expression via Mutalyzer 3.

    Best-effort and offline-safe: returns ``{"ok": False, "error": ...}`` on
    any failure so callers can degrade gracefully.
    """
    if not hgvs:
        return {"ok": False, "error": "empty hgvs"}
    try:
        res = get_json(f"{MUTALYZER}/normalize/{hgvs}")
        # Mutalyzer 3 returns a `normalized_description` on success and an
        # `errors` / `messages` list on failure.
        if res.get("normalized_description"):
            return {
                "ok": True,
                "input": hgvs,
                "normalized": res.get("normalized_description"),
                "protein": _safe(res, "protein", "description"),
                "equivalent": [e.get("description") for e in (res.get("equivalent_descriptions") or [])][:5],
            }
        problems = res.get("errors") or res.get("messages") or []
        return {"ok": False, "input": hgvs, "errors": problems[:5]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "input": hgvs, "error": str(e)}
