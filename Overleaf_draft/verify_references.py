#!/usr/bin/env python3
"""
Reference Verification Script for Academic Papers
==================================================
Detects hallucinated, broken, or mismatched references in .bib files.

Checks:
  1. URL reachability (HTTP status codes, redirects, 404s)
  2. Semantic Scholar API cross-check (title match, year, venue, authors)
  3. arXiv metadata validation (for arxiv.org URLs)
  4. Flags: fabricated papers, wrong year, wrong venue, wrong authors, dead URLs

Usage:
  # Real-time streaming to terminal + log file:
  python3 -u Overleaf_draft/verify_references.py Overleaf_draft/version_EMNLP/custom.bib 2>&1 | tee logs/verification_report.log

  # Background run (survives terminal close, check logs later):
  cd /Users/kapilwanaskar/Downloads/research_projects/finetuning_evaluation
  nohup python3 -u Overleaf_draft/verify_references.py Overleaf_draft/version_EMNLP/custom.bib > logs/verification_report.log 2>&1 &
  # Then check: jobs -l   OR   tail -f logs/verification_report.log
"""

import argparse
import io
import re
import sys
import time
import json
import urllib.request
import urllib.error
import urllib.parse
import ssl
from pathlib import Path
from collections import defaultdict

# Force unbuffered stdout so output streams in real-time (even when piped to tee)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)


# ─────────────────────────────────────────────────────────────────────────────
# COLORS for terminal output
# ─────────────────────────────────────────────────────────────────────────────
class C:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# BIB PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_bib_file(bib_path):
    """Parse a .bib file and extract entries with key fields."""
    with open(bib_path, "r", encoding="utf-8") as f:
        content = f.read()

    entries = []
    # Match @type{key, ... }
    pattern = re.compile(
        r"@(\w+)\s*\{([^,]+),\s*(.*?)\n\}",
        re.DOTALL,
    )

    for match in pattern.finditer(content):
        entry_type = match.group(1).lower()
        cite_key = match.group(2).strip()
        body = match.group(3)

        # Extract fields
        fields = {}
        # Match field = {value} or field = "value" or field = number
        field_pattern = re.compile(
            r"(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\"|(\d+))"
        )
        for fm in field_pattern.finditer(body):
            fname = fm.group(1).lower()
            fval = fm.group(2) or fm.group(3) or fm.group(4)
            if fval:
                # Clean LaTeX commands
                fval = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", fval)
                fval = re.sub(r"[{}]", "", fval)
                fval = fval.strip()
                fields[fname] = fval

        entries.append(
            {
                "type": entry_type,
                "key": cite_key,
                "title": fields.get("title", ""),
                "author": fields.get("author", ""),
                "year": fields.get("year", ""),
                "journal": fields.get("journal", ""),
                "booktitle": fields.get("booktitle", ""),
                "url": fields.get("url", ""),
                "volume": fields.get("volume", ""),
                "pages": fields.get("pages", ""),
                "doi": fields.get("doi", ""),
            }
        )

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# URL VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def check_url(url, timeout=15):
    """Check if a URL is reachable. Returns (status_code, final_url, error_msg)."""
    if not url:
        return None, None, "NO_URL"

    try:
        # Create SSL context that doesn't verify (some academic sites have cert issues)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Academic Reference Checker)"
            },
        )
        response = urllib.request.urlopen(req, timeout=timeout, context=ctx)
        return response.getcode(), response.geturl(), None
    except urllib.error.HTTPError as e:
        return e.code, url, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return None, url, f"URL Error: {e.reason}"
    except Exception as e:
        return None, url, f"Error: {str(e)[:80]}"


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC SCHOLAR API
# ─────────────────────────────────────────────────────────────────────────────
def query_semantic_scholar(title, max_retries=4):
    """Query Semantic Scholar API by title. Returns paper metadata or None."""
    if not title:
        return None

    # Clean title for search
    clean_title = re.sub(r"[^\w\s]", " ", title).strip()
    clean_title = re.sub(r"\s+", " ", clean_title)

    encoded = urllib.parse.quote(clean_title)
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit=3&fields=title,year,venue,authors,externalIds,publicationVenue"

    for attempt in range(max_retries):
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(
                api_url,
                headers={"User-Agent": "Academic-Ref-Checker/1.0"},
            )
            response = urllib.request.urlopen(req, timeout=15, context=ctx)
            data = json.loads(response.read().decode("utf-8"))

            if data.get("data") and len(data["data"]) > 0:
                return data["data"]
            return None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited — wait and retry
                wait = 3 * (attempt + 1)
                print(
                    f"    {C.YELLOW}[Rate limited by Semantic Scholar, waiting {wait}s...]{C.END}"
                )
                time.sleep(wait)
                continue
            return None
        except Exception:
            return None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# ARXIV METADATA CHECK
# ─────────────────────────────────────────────────────────────────────────────
def extract_arxiv_id(url):
    """Extract arXiv ID from URL."""
    if not url:
        return None
    match = re.search(r"arxiv\.org/abs/(\d+\.\d+)", url)
    if match:
        return match.group(1)
    return None


def query_arxiv(arxiv_id):
    """Query arXiv API for paper metadata."""
    if not arxiv_id:
        return None
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "Academic-Ref-Checker/1.0"},
        )
        response = urllib.request.urlopen(req, timeout=15, context=ctx)
        xml_data = response.read().decode("utf-8")

        # Parse title
        title_match = re.search(r"<title>(.*?)</title>", xml_data, re.DOTALL)
        # Parse authors
        authors = re.findall(r"<name>(.*?)</name>", xml_data)
        # Parse published year
        published_match = re.search(r"<published>(\d{4})", xml_data)

        if title_match:
            # Skip the feed title (first <title> is the feed)
            titles = re.findall(r"<title>(.*?)</title>", xml_data, re.DOTALL)
            paper_title = titles[-1].strip() if len(titles) > 1 else titles[0].strip()
            paper_title = re.sub(r"\s+", " ", paper_title)
            return {
                "title": paper_title,
                "authors": authors,
                "year": published_match.group(1) if published_match else None,
            }
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def normalize_title(title):
    """Normalize title for comparison."""
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_similarity(t1, t2):
    """Simple word-overlap similarity between two titles."""
    if not t1 or not t2:
        return 0.0
    w1 = set(normalize_title(t1).split())
    w2 = set(normalize_title(t2).split())
    if not w1 or not w2:
        return 0.0
    intersection = w1 & w2
    union = w1 | w2
    return len(intersection) / len(union)


def extract_last_names(author_str):
    """Extract last names from bib author string."""
    if not author_str:
        return set()
    # Split by "and"
    authors = re.split(r"\s+and\s+", author_str)
    last_names = set()
    for a in authors:
        a = a.strip()
        if not a or a == "others":
            continue
        # "Last, First" or "First Last"
        if "," in a:
            last = a.split(",")[0].strip()
        else:
            parts = a.split()
            last = parts[-1] if parts else a
        last_names.add(last.lower())
    return last_names


def compare_authors(bib_authors, api_authors):
    """Compare author lists. Returns overlap ratio."""
    bib_names = extract_last_names(bib_authors)
    if not bib_names:
        return 1.0  # Can't check, assume OK

    api_names = set()
    for a in api_authors:
        name = a.get("name", "") if isinstance(a, dict) else str(a)
        parts = name.split()
        if parts:
            api_names.add(parts[-1].lower())

    if not api_names:
        return 1.0  # Can't check

    intersection = bib_names & api_names
    # Use the smaller set as denominator (bib may use "others")
    denom = min(len(bib_names), len(api_names))
    if denom == 0:
        return 1.0
    return len(intersection) / denom


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def verify_entry(entry, idx, total):
    """Verify a single bib entry. Returns list of issues."""
    issues = []
    key = entry["key"]
    title = entry["title"]
    year = entry["year"]
    url = entry["url"]

    print(
        f"\n{C.BOLD}[{idx}/{total}] {C.CYAN}{key}{C.END}"
        f"  {C.BOLD}{title[:70]}{'...' if len(title) > 70 else ''}{C.END}"
    )

    # ── CHECK 1: URL reachability ──
    if url:
        status, final_url, err = check_url(url)
        if status and 200 <= status < 400:
            print(f"  {C.GREEN}[URL OK]{C.END} {status} — {url[:80]}")
        elif status == 404:
            msg = f"BROKEN URL (404 Not Found): {url}"
            issues.append(("BROKEN_URL", msg))
            print(f"  {C.RED}[BROKEN URL] 404 Not Found{C.END} — {url}")
        elif status and status >= 400:
            msg = f"URL ERROR ({status}): {url}"
            issues.append(("URL_ERROR", msg))
            print(f"  {C.YELLOW}[URL ERROR] HTTP {status}{C.END} — {url}")
        elif err:
            msg = f"URL UNREACHABLE: {url} — {err}"
            issues.append(("URL_UNREACHABLE", msg))
            print(f"  {C.YELLOW}[URL UNREACHABLE]{C.END} {err} — {url[:80]}")
    else:
        issues.append(("NO_URL", f"No URL provided for {key}"))
        print(f"  {C.YELLOW}[NO URL]{C.END} No URL to verify")

    # ── CHECK 2: arXiv metadata (if arXiv URL) ──
    arxiv_id = extract_arxiv_id(url)
    if arxiv_id:
        arxiv_data = query_arxiv(arxiv_id)
        if arxiv_data:
            # Title match
            sim = title_similarity(title, arxiv_data["title"])
            if sim < 0.5:
                msg = (
                    f"ARXIV TITLE MISMATCH: bib='{title}' vs "
                    f"arxiv='{arxiv_data['title']}' (similarity={sim:.2f})"
                )
                issues.append(("ARXIV_TITLE_MISMATCH", msg))
                print(
                    f"  {C.RED}[ARXIV TITLE MISMATCH]{C.END} "
                    f"similarity={sim:.2f}\n"
                    f"    bib:   {title[:80]}\n"
                    f"    arxiv: {arxiv_data['title'][:80]}"
                )
            else:
                print(f"  {C.GREEN}[arXiv title match]{C.END} similarity={sim:.2f}")

            # Year check
            if arxiv_data.get("year") and year:
                if abs(int(arxiv_data["year"]) - int(year)) > 1:
                    msg = (
                        f"ARXIV YEAR MISMATCH: bib={year} vs "
                        f"arxiv={arxiv_data['year']}"
                    )
                    issues.append(("ARXIV_YEAR_MISMATCH", msg))
                    print(
                        f"  {C.YELLOW}[ARXIV YEAR MISMATCH]{C.END} "
                        f"bib={year} vs arxiv={arxiv_data['year']}"
                    )
        time.sleep(0.3)  # Be polite to arXiv API

    # ── CHECK 3: Semantic Scholar cross-check ──
    time.sleep(1.5)  # Rate limit: unauthenticated users share a single key
    ss_results = query_semantic_scholar(title)

    if ss_results:
        # Find best matching paper
        best_match = None
        best_sim = 0
        for paper in ss_results:
            sim = title_similarity(title, paper.get("title", ""))
            if sim > best_sim:
                best_sim = sim
                best_match = paper

        if best_match and best_sim >= 0.6:
            print(
                f"  {C.GREEN}[Semantic Scholar FOUND]{C.END} "
                f"similarity={best_sim:.2f}"
            )

            # Year comparison
            ss_year = best_match.get("year")
            if ss_year and year:
                try:
                    if abs(int(ss_year) - int(year)) > 1:
                        msg = (
                            f"YEAR MISMATCH: bib={year} vs "
                            f"Semantic Scholar={ss_year}"
                        )
                        issues.append(("YEAR_MISMATCH", msg))
                        print(
                            f"  {C.YELLOW}[YEAR MISMATCH]{C.END} "
                            f"bib={year} vs S2={ss_year}"
                        )
                except ValueError:
                    pass

            # Venue comparison
            ss_venue = best_match.get("venue", "")
            pub_venue = best_match.get("publicationVenue")
            if pub_venue and isinstance(pub_venue, dict):
                ss_venue = pub_venue.get("name", ss_venue)

            bib_venue = entry["journal"] or entry["booktitle"]
            if ss_venue and bib_venue:
                venue_sim = title_similarity(bib_venue, ss_venue)
                if venue_sim < 0.2 and len(ss_venue) > 3:
                    print(
                        f"  {C.CYAN}[VENUE INFO]{C.END} "
                        f"bib='{bib_venue[:50]}' vs S2='{ss_venue[:50]}'"
                    )

            # Author comparison
            ss_authors = best_match.get("authors", [])
            if ss_authors:
                author_overlap = compare_authors(entry["author"], ss_authors)
                if author_overlap < 0.3:
                    bib_names = extract_last_names(entry["author"])
                    api_names = set()
                    for a in ss_authors:
                        name = (
                            a.get("name", "") if isinstance(a, dict) else str(a)
                        )
                        parts = name.split()
                        if parts:
                            api_names.add(parts[-1].lower())
                    msg = (
                        f"AUTHOR MISMATCH: bib={bib_names} vs "
                        f"S2={api_names} (overlap={author_overlap:.2f})"
                    )
                    issues.append(("AUTHOR_MISMATCH", msg))
                    print(
                        f"  {C.RED}[AUTHOR MISMATCH]{C.END} "
                        f"overlap={author_overlap:.2f}\n"
                        f"    bib: {list(bib_names)[:5]}\n"
                        f"    S2:  {list(api_names)[:5]}"
                    )

        elif best_sim < 0.4:
            msg = (
                f"NOT FOUND on Semantic Scholar: '{title}' "
                f"(best match similarity={best_sim:.2f})"
            )
            issues.append(("NOT_FOUND_S2", msg))
            print(
                f"  {C.RED}[NOT FOUND on Semantic Scholar]{C.END} "
                f"best similarity={best_sim:.2f}"
            )
            if best_match:
                print(
                    f"    closest: '{best_match.get('title', 'N/A')[:80]}'"
                )
        else:
            print(
                f"  {C.YELLOW}[PARTIAL MATCH on S2]{C.END} "
                f"similarity={best_sim:.2f}"
            )
    else:
        msg = f"NO RESULTS from Semantic Scholar for: '{title}'"
        issues.append(("S2_NO_RESULTS", msg))
        print(f"  {C.YELLOW}[S2 NO RESULTS]{C.END} Could not query/find paper")

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_report(all_issues, total_entries):
    """Print a categorized summary report."""
    print("\n" + "=" * 80)
    print(f"{C.BOLD}REFERENCE VERIFICATION REPORT{C.END}")
    print("=" * 80)

    # Categorize
    categories = defaultdict(list)
    for key, issue_type, msg in all_issues:
        categories[issue_type].append((key, msg))

    # Severity ordering
    severity_order = [
        ("BROKEN_URL", "BROKEN URLs (404)", C.RED),
        ("ARXIV_TITLE_MISMATCH", "arXiv TITLE MISMATCHES (possible hallucination)", C.RED),
        ("NOT_FOUND_S2", "NOT FOUND on Semantic Scholar (possible hallucination)", C.RED),
        ("AUTHOR_MISMATCH", "AUTHOR MISMATCHES", C.RED),
        ("YEAR_MISMATCH", "YEAR MISMATCHES", C.YELLOW),
        ("ARXIV_YEAR_MISMATCH", "arXiv YEAR MISMATCHES", C.YELLOW),
        ("URL_ERROR", "URL ERRORS (non-404)", C.YELLOW),
        ("URL_UNREACHABLE", "URL UNREACHABLE", C.YELLOW),
        ("S2_NO_RESULTS", "Semantic Scholar returned NO RESULTS", C.YELLOW),
        ("NO_URL", "MISSING URLs", C.CYAN),
    ]

    critical_count = 0
    warning_count = 0

    for issue_type, label, color in severity_order:
        if issue_type in categories:
            items = categories[issue_type]
            is_critical = color == C.RED
            if is_critical:
                critical_count += len(items)
            else:
                warning_count += len(items)

            print(f"\n{color}{C.BOLD}{'!'*3} {label} ({len(items)}){C.END}")
            print("-" * 60)
            for key, msg in items:
                print(f"  {C.BOLD}{key}{C.END}: {msg}")

    # Summary
    clean_count = total_entries - len(
        set(key for key, _, _ in all_issues)
    )
    print("\n" + "=" * 80)
    print(f"{C.BOLD}SUMMARY{C.END}")
    print(f"  Total references checked:  {total_entries}")
    print(f"  {C.GREEN}Clean (no issues):           {clean_count}{C.END}")
    print(f"  {C.RED}Critical (likely problems):   {critical_count}{C.END}")
    print(f"  {C.YELLOW}Warnings (review manually):  {warning_count}{C.END}")
    print("=" * 80)

    if critical_count > 0:
        print(
            f"\n{C.RED}{C.BOLD}ACTION REQUIRED: {critical_count} references "
            f"need manual verification before submission!{C.END}"
        )
        print(
            f"  Hallucinated references → DESK REJECTION at ACL/EMNLP/ARR\n"
        )
    elif warning_count > 0:
        print(
            f"\n{C.YELLOW}Some warnings found. Review them but likely OK.{C.END}\n"
        )
    else:
        print(f"\n{C.GREEN}{C.BOLD}All references look clean!{C.END}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Verify .bib references against Semantic Scholar and URL checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_references.py version_EMNLP/custom.bib
  python verify_references.py version_EMNLP/custom.bib version_ACL/refs.bib
  python verify_references.py --skip-urls version_EMNLP/custom.bib
        """,
    )
    parser.add_argument(
        "bib_files",
        nargs="+",
        help="Path(s) to .bib file(s) to verify",
    )
    parser.add_argument(
        "--skip-urls",
        action="store_true",
        help="Skip URL reachability checks (faster)",
    )
    parser.add_argument(
        "--skip-s2",
        action="store_true",
        help="Skip Semantic Scholar API checks",
    )
    parser.add_argument(
        "--skip-arxiv",
        action="store_true",
        help="Skip arXiv metadata checks",
    )

    args = parser.parse_args()

    all_issues = []
    total_entries = 0

    for bib_path in args.bib_files:
        path = Path(bib_path)
        if not path.exists():
            print(f"{C.RED}ERROR: File not found: {bib_path}{C.END}")
            sys.exit(1)

        print(f"\n{'='*80}")
        print(f"{C.BOLD}Verifying: {path}{C.END}")
        print(f"{'='*80}")

        entries = parse_bib_file(path)
        print(f"Found {len(entries)} references\n")

        for i, entry in enumerate(entries, 1):
            # Respect flags
            if args.skip_urls:
                entry["_skip_url"] = True
            if args.skip_s2:
                entry["_skip_s2"] = True
            if args.skip_arxiv:
                entry["_skip_arxiv"] = True

            issues = verify_entry(entry, i, len(entries))
            for issue_type, msg in issues:
                all_issues.append((entry["key"], issue_type, msg))

            total_entries += 1

    print_report(all_issues, total_entries)


if __name__ == "__main__":
    main()
