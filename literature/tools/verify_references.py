"""Phase 02 - verify every candidate reference against Crossref / arXiv / OpenAlex and emit BibTeX.

Nothing is cited in the manuscript unless it appears here with status VERIFIED.

Usage (from article-01-seattle/):
    python literature/tools/verify_references.py literature/references_metadata.json literature/references.bib
"""
import html
import json
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request

try:  # Windows hosts behind TLS interception: use the OS trust store when available.
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

UA = {"User-Agent": "article01-refcheck/1.0 (mailto:research@example.invalid)"}

# key -> DOI or "arXiv:<id>"
CANDIDATES = {
    # Seattle benchmarking data
    "pan2020data": "10.1016/j.apenergy.2020.114965",
    "zhang2023xai": "10.1016/j.energy.2022.125468",
    "zhang2023gtwr": "10.1016/j.energy.2023.126747",
    "worthy2025eo": "10.1016/j.scs.2025.106544",
    "arjunan2020energystarpp": "10.1016/j.apenergy.2020.115413",
    "arjunan2019energystarpp_arxiv": "arXiv:1910.14563",
    "yap2025revealing": "10.1038/s41893-025-01615-8",
    "li2024generalized": "10.1016/j.dibe.2024.100320",
    "duan2026ghgbench": "arXiv:2605.13743",
    # City-scale benchmarking prediction and data quality
    "kontokosta2017citywide": "10.1016/j.apenergy.2017.04.005",
    "papadopoulos2019grading": "10.1016/j.apenergy.2018.10.053",
    "hsu2014selfreported": "10.1080/09613218.2014.887612",
    "robinson2017commercial": "10.1016/j.apenergy.2017.09.060",
    "amasyali2018review": "10.1016/j.rser.2017.04.095",
    # Building energy datasets / benchmarks
    "miller2020bdg2": "10.1038/s41597-020-00712-x",
    "miller2020gepiii": "10.1080/23744731.2020.1795514",
    "emami2023buildingsbench": "arXiv:2307.00142",
    # Evaluation methodology, leakage, uncertainty
    "kapoor2023leakage": "10.1016/j.patter.2023.100804",
    "kaufman2012leakage": "10.1145/2382577.2382579",
    "roberts2017cv": "10.1111/ecog.02881",
    "bergmeir2012cvts": "10.1016/j.ins.2011.12.028",
    "varma2006bias": "10.1186/1471-2105-7-91",
    "duan1983smearing": "10.1080/01621459.1983.10478017",
    "field2007clustered": "10.1111/j.1467-9868.2007.00593.x",
    "roth2026leakagetypes": "arXiv:2604.04199",
    # Data versioning / revisions
    "croushore2001realtime": "10.1016/S0304-4076(01)00072-0",
    "gebru2021datasheets": "10.1145/3458723",
    # Resolved by exact-title search (DOI not known in advance)
    "papadopoulos2018pattern": "title:Pattern recognition in building energy performance over time using energy benchmarking data",
    "arjunan2022beem": "title:BEEM: Data-driven building energy benchmarking for Singapore",
    "konhauser2022ensemble": "title:Leveraging advanced ensemble models to increase building energy performance prediction accuracy in the residential building sector",
}

# Grey literature and data sources without a DOI: the URL was opened and checked during Phase 02.
GREY = {
    "scofield2014energystar": """@inproceedings{scofield2014energystar,
  title = {{{ENERGY STAR} Building Benchmarking Scores: Good Idea, Bad Science}},
  author = {Scofield, John H.},
  booktitle = {Proceedings of the 2014 ACEEE Summer Study on Energy Efficiency in Buildings},
  year = {2014},
  url = {https://www.aceee.org/files/proceedings/2014/data/papers/3-725.pdf}
}""",
    "epa2021energystarscore": """@techreport{epa2021energystarscore,
  title = {{ENERGY STAR} Score Technical Reference},
  author = {{U.S. Environmental Protection Agency}},
  institution = {ENERGY STAR Portfolio Manager},
  year = {2021},
  month = apr,
  url = {https://portfoliomanager.energystar.gov/pdf/reference/ENERGY%20STAR%20Score.pdf}
}""",
    "seattle2026benchmarking": """@misc{seattle2026benchmarking,
  title = {{Building Energy Benchmarking Data, 2015--Present (dataset teqw-tu6e)}},
  author = {{City of Seattle Office of Sustainability and Environment}},
  year = {2026},
  note = {Public Domain. Snapshot downloaded 2026-10-29, SHA-256 d1a8227bf9c7c82aea8279a67405add617271852a985d96bb6e4ac9070e8250f},
  url = {https://data.seattle.gov/Built-Environment/Building-Energy-Benchmarking-Data-2015-Present/teqw-tu6e}
}""",
    "seattle2016benchmarking_original": """@misc{seattle2016benchmarking_original,
  title = {{2016 Building Energy Benchmarking (original annual release, former dataset 2bpz-gwpy)}},
  author = {{City of Seattle Office of Sustainability and Environment}},
  year = {2017},
  note = {No longer served by data.seattle.gov (checked 2026-10-29). Copy distributed by OpenClassrooms, SHA-256 8bcb3dc96494f7c02c2e5b888d71ea4ad451b22657a7243d37de3573d8923a9d},
  url = {https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/Data_Scientist_P4/2016_Building_Energy_Benchmarking.csv}
}""",
}


def latex_safe(bib):
    """Make Crossref BibTeX compile: decode HTML entities, escape bare '&', normalise Unicode dashes."""
    bib = html.unescape(bib)
    bib = re.sub(r"(?<!\\)&", r"\\&", bib)
    bib = bib.replace("–", "--").replace("—", "--").replace("‐", "-").replace("‑", "-")
    # Keep the publisher's capitalisation (acronyms, place names): BibTeX styles lower-case unprotected titles.
    bib = re.sub(r"title=\{(?!\{)(.*?)\}, (?=\w+=)", lambda m: "title={{" + m.group(1) + "}}, ", bib, count=1)
    # The DOI already identifies the work: drop the redundant dx.doi.org URL and the ISSN.
    if "DOI={" in bib:
        bib = re.sub(r"\s*url=\{http://dx\.doi\.org/[^}]*\},", "", bib)
    bib = re.sub(r"\s*ISSN=\{[^}]*\},", "", bib)
    return bib


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def doi_from_title(title):
    q = urllib.parse.quote(title)
    items = json.loads(get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=5"))["message"]["items"]
    for it in items:
        if norm((it.get("title") or [""])[0]) == norm(title):
            return it["DOI"]
    raise LookupError("no exact-title match in Crossref")


def get(url, accept=None):
    req = urllib.request.Request(url, headers={**UA, **({"Accept": accept} if accept else {})})
    with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context()) as r:
        return r.read().decode("utf-8", errors="replace")


def openalex(doi):
    try:
        w = json.loads(get("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi)))
    except Exception:
        return {}
    abstract = ""
    inv = w.get("abstract_inverted_index") or {}
    if inv:
        pos = sorted((p, word) for word, ps in inv.items() for p in ps)
        abstract = " ".join(word for _, word in pos)
    oa = w.get("open_access") or {}
    return {"openalex_id": w.get("id"), "cited_by_count": w.get("cited_by_count"), "is_oa": oa.get("is_oa"),
            "oa_url": oa.get("oa_url"), "abstract": abstract}


def crossref(doi):
    m = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    authors = [f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in m.get("author", [])]
    rec = {"status": "VERIFIED", "source": "crossref", "doi": m.get("DOI"), "title": (m.get("title") or [""])[0],
           "authors": authors, "venue": (m.get("container-title") or [""])[0], "volume": m.get("volume"),
           "pages_or_article": m.get("article-number") or m.get("page"),
           "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0], "type": m.get("type")}
    try:
        rec["bibtex"] = get("https://api.crossref.org/works/" + urllib.parse.quote(doi) + "/transform/application/x-bibtex")
    except Exception as e:  # BibTeX transform is occasionally unavailable
        rec["bibtex"] = ""
        rec["bibtex_error"] = str(e)
    rec.update(openalex(doi))
    return rec


def arxiv(aid):
    x = get("https://export.arxiv.org/api/query?id_list=" + aid)
    titles = re.findall(r"<title>(.*?)</title>", x, re.S)
    if len(titles) < 2:
        return {"status": "NOT_FOUND", "source": "arxiv", "arxiv": aid}
    title = " ".join(titles[1].split())
    authors = re.findall(r"<name>(.*?)</name>", x)
    published = re.findall(r"<published>(.*?)</published>", x)[0]
    summary = " ".join((re.findall(r"<summary>(.*?)</summary>", x, re.S) or [""])[0].split())
    jref = re.findall(r"<arxiv:journal_ref[^>]*>(.*?)<", x)
    key_year = published[:4]
    bib = ("@misc{%s,\n  title = {{%s}},\n  author = {%s},\n  year = {%s},\n  eprint = {%s},\n"
           "  archivePrefix = {arXiv},\n  url = {https://arxiv.org/abs/%s}\n}" %
           ("{key}", title, " and ".join(authors), key_year, aid, aid))
    return {"status": "VERIFIED", "source": "arxiv", "arxiv": aid, "title": title, "authors": authors,
            "year": int(key_year), "published": published, "journal_ref": jref[0] if jref else None,
            "abstract": summary, "bibtex": bib}


def main(json_out, bib_out):
    out, bibs = {}, []
    for key, ident in CANDIDATES.items():
        try:
            if ident.startswith("arXiv:"):
                rec = arxiv(ident[6:])
            elif ident.startswith("title:"):
                rec = crossref(doi_from_title(ident[6:]))
            else:
                rec = crossref(ident)
        except Exception as e:
            rec = {"status": "NOT_FOUND", "identifier": ident, "error": str(e)}
        rec["key"] = key
        out[key] = rec
        if rec.get("bibtex"):
            b = rec["bibtex"].replace("{key}", key)
            b = re.sub(r"^(@\w+\{)[^,]*,", r"\g<1>" + key + ",", b.strip(), count=1)
            bibs.append(latex_safe(b))
        print(f"{rec['status']:9s} {key:32s} {rec.get('year')} | {(rec.get('title') or '')[:90]}")
        time.sleep(0.3)
    for key, bib in GREY.items():
        out[key] = {"key": key, "status": "VERIFIED_GREY", "source": "manual (URL checked)",
                    "url": re.search(r"url = \{(.*?)\}", bib).group(1)}
        bibs.append(bib)
        print(f"{'GREY':9s} {key}")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(bib_out, "w", encoding="utf-8") as f:
        f.write("% Generated by literature/tools/verify_references.py from Crossref/arXiv metadata.\n\n")
        f.write("\n\n".join(bibs) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
