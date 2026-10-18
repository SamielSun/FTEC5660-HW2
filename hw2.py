#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


# --- Task 1: student helpers -------------------------------------------------
# Everything in this block sits above the two required functions. The provided
# runner code below ``score_cvs`` is not touched.

# Returned when a CV cannot be verified at all (parse failure, tool error, ...).
FALLBACK_SCORE = 0.05
# How many LinkedIn candidates we are willing to fetch before deciding.
MAX_CANDIDATES = 8
# Attempts per CV when the model call or the tool loop fails transiently.
AGENT_ATTEMPTS = 3
# A candidate whose signal score reaches this is taken as the identity match.
CONFIDENT_IDENTITY_SCORE = 6.0

# Fields the assignment classifies as possible discrepancies. The headline,
# the hometown and the job-description bullets are deliberately excluded.
VERIFIER_SYSTEM_PROMPT = """You are a CV verification agent for KYC-style background checks.

A candidate's CV is being checked against their public LinkedIn profile. All
evidence must come from the tools you are given; never use outside knowledge.

IDENTITY
Names in this dataset are shared by hundreds of people. A first-pass lookup has
already proposed a LinkedIn `person_id`, but you must confirm it yourself: call
`get_linkedin_profile` and check that the CV's employers, schools and skills
really overlap with that profile. If it is clearly a different person, use
`search_linkedin_people` (narrow with `location` and `industry`) to find the
right one instead.

FIELDS YOU MAY FLAG — nothing else
  * name
  * city
  * jobs: company, title, seniority, start year, end year
  * education: degree, school, field, graduation year
  * skills

NEVER treat any of these as a discrepancy:
  * the headline / professional title line
  * the hometown
  * the job-description bullet points
  * a CV that simply says less than the profile

NOT a discrepancy (wording or detail only):
  * "Bachelor of Science" vs BSc, "MBA" vs "Master of Business Administration"
  * "UI/UX Design" vs UI/UX
  * "Senior Manager" for a role whose title is "Manager" and seniority "senior"
  * a CV that lists FEWER skills than the profile
  * a CV that omits seniority, month precision or an optional field

IS a discrepancy (any single one is enough):
  * an inflated or downgraded title
  * a start or end year that differs, even by one
  * an upgraded or invented degree, school or employer
  * a skill on the CV that the profile does not contain
  * a wrong city or a wrong name

OUTPUT FORMAT
After you finish checking, end your reply with exactly one line of JSON:
{"score": <number between 0 and 1>, "reasons": ["<short reason>", ...]}
The score must be greater than 0.5 when every comparable field agrees, and 0.5
or lower as soon as one real discrepancy is found. Use the whole range to
express how confident you are."""

_SECTION_ALIASES = {
    "professional experience": "jobs",
    "work experience": "jobs",
    "experience": "jobs",
    "employment": "jobs",
    "education": "education",
    "skills": "skills",
    "technical skills": "skills",
    "core skills": "skills",
}

_YEAR = r"(?:19|20)\d{2}"

# "Senior Manager, Deloitte    2020 <sep> Present"  (the separator may be a
# mangled en-dash such as "\ufffdC" once markitdown has extracted the PDF).
_JOB_LINE = re.compile(
    r"^(?P<title>[^,\n]{2,80}?)\s*,\s*(?P<company>[^\n]+?)\s+"
    r"(?P<start>" + _YEAR + r")\s*[^\n\d]{0,14}?"
    r"(?P<end>" + _YEAR + r"|present|current|now)\s*$",
    re.IGNORECASE,
)

# "PhD in AI    Graduated 2004", or the table-layout variant "BSc in Design 2011"
_EDU_LINE = re.compile(
    r"^(?P<degree>[^\n]{2,90}?)\s+(?:Graduated\s*)?(?P<year>" + _YEAR + r")\s*$",
    re.IGNORECASE,
)

# "London, UK | Hometown: Stanford"  or  "Munich, Germany"
_LOCATION_LINE = re.compile(
    r"^(?P<city>[^|,\n]+?)\s*,\s*(?P<country>[^|\n]+?)\s*(?:\|[^\n]*)?$",
    re.IGNORECASE,
)

_BULLETS = ("-", "*", "\u2022", "\u00b7", "\u25aa", "\u25cf")
_TABLE_RULE_ROW = re.compile(r"^[-: ]+$")


def _norm(value: Any) -> str:
    """Lower-case a value and drop everything that is not alphanumeric."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tool_payload(result: Any) -> Any:
    """Unwrap the ``[{"type": "text", "text": "<JSON>"}]`` shape of MCP tools."""
    if isinstance(result, list):
        chunks: list[Any] = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                try:
                    chunks.append(json.loads(block["text"]))
                except (TypeError, ValueError):
                    chunks.append(block.get("text"))
            else:
                chunks.append(block)
        return chunks[0] if len(chunks) == 1 else chunks
    return result


def _flatten_table_rows(text: str) -> str:
    """Collapse markdown table rows into plain lines (some CV PDFs are tables)."""
    out: list[str] = []
    for line in text.replace("\r", "").split("\n"):
        if line.count("|") >= 2:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            cells = [cell for cell in cells if cell]
            if not cells or all(_TABLE_RULE_ROW.match(cell) for cell in cells):
                continue
            line = " ".join(cells)
        out.append(line)
    return "\n".join(out)


def _cv_claims(text: str) -> dict[str, Any]:
    """Regex-parse the generated CV layout into the fields we can verify."""
    text = _flatten_table_rows(text)
    claims: dict[str, Any] = {
        "name": "",
        "headline": "",
        "city": "",
        "country": "",
        "jobs": [],
        "education": [],
        "schools": [],
        "skills": [],
    }
    lines = [line.strip() for line in text.replace("\r", "").split("\n")]
    filled = [line for line in lines if line]

    if filled:
        claims["name"] = filled[0]
    if len(filled) > 1:
        claims["headline"] = filled[1]
    for candidate in filled[1:3]:
        match = _LOCATION_LINE.match(candidate)
        if match:
            claims["city"] = match.group("city").strip()
            claims["country"] = match.group("country").strip()
            break

    section: str | None = None
    pending_education: dict[str, Any] | None = None
    pending_school = False

    for raw in lines:
        if not raw:
            continue
        lowered = raw.lower().rstrip(":")
        if lowered in _SECTION_ALIASES:
            section = _SECTION_ALIASES[lowered]
            pending_school = False
            pending_education = None
            continue
        bullet = raw.lstrip("".join(_BULLETS) + " ").strip()
        if not bullet or bullet.isdigit():
            continue  # empty bullet, or a page number from the PDF footer

        if section == "jobs":
            if raw[0] in _BULLETS:
                continue  # a responsibility bullet is never a claim
            match = _JOB_LINE.match(raw)
            if match:
                end = match.group("end").lower()
                claims["jobs"].append(
                    {
                        "title": match.group("title").strip(),
                        "company": match.group("company").strip(),
                        "start": int(match.group("start")),
                        "end": None
                        if end in ("present", "current", "now")
                        else int(match.group("end")),
                    }
                )
        elif section == "education":
            match = _EDU_LINE.match(raw)
            if match:
                degree = match.group("degree").strip()
                field = ""
                if " in " in degree:
                    degree, field = degree.rsplit(" in ", 1)
                pending_education = {
                    "degree": degree.strip(),
                    "field": field.strip(),
                    "year": int(match.group("year")),
                    "school": "",
                }
                pending_school = True
            elif pending_school and pending_education is not None:
                pending_education["school"] = raw
                claims["education"].append(pending_education)
                claims["schools"].append(raw)
                pending_education = None
                pending_school = False
        elif section == "skills":
            claims["skills"].extend(
                part.strip() for part in bullet.split(",") if part.strip()
            )

    return claims


def _industry_hint(claims: dict[str, Any]) -> str:
    """Derive an industry filter from the CV headline, e.g. 'AI Professional'."""
    headline = claims.get("headline") or ""
    cleaned = re.sub(r"\bprofessional\b", " ", headline, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^A-Za-z& ]+", " ", cleaned)
    words = [word for word in cleaned.split() if len(word) > 1]
    if not words:
        return ""
    return words[0] if len(words) == 1 else " ".join(words[:2])


def _signal_score(claims: dict[str, Any], profile: dict[str, Any]) -> float:
    """How strongly a LinkedIn profile overlaps with what the CV claims."""
    score = 0.0
    companies = {_norm(job.get("company")) for job in claims.get("jobs", [])}
    profile_companies = {
        _norm(job.get("company")) for job in profile.get("experience", [])
    }
    score += 3.0 * len((companies - {""}) & (profile_companies - {""}))

    schools = {_norm(name) for name in claims.get("schools", [])}
    profile_schools = {
        _norm(entry.get("school")) for entry in profile.get("education", [])
    }
    score += 3.0 * len((schools - {""}) & (profile_schools - {""}))

    skills = {_norm(name) for name in claims.get("skills", [])} - {""}
    profile_skills = {
        _norm(entry.get("name")) for entry in profile.get("skills", [])
    }
    if skills:
        score += 2.0 * len(skills & (profile_skills - {""})) / len(skills)

    city = _norm(claims.get("city"))
    if city and city == _norm(profile.get("city")):
        score += 1.0
    return score


async def _resolve_person(
    tools: dict[str, Any], claims: dict[str, Any]
) -> dict[str, Any] | None:
    """Find the LinkedIn profile that most likely belongs to this CV."""
    search = tools.get("search_linkedin_people")
    lookup = tools.get("get_linkedin_profile")
    if search is None or lookup is None:
        return None

    name = (claims.get("name") or "").strip()
    if not name:
        return None
    city = (claims.get("city") or "").strip()
    industry = _industry_hint(claims)

    queries: list[dict[str, Any]] = []
    if city and industry:
        queries.append({"q": name, "location": city, "industry": industry})
    if city:
        queries.append({"q": name, "location": city})
    queries.append({"q": name})

    ordered: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for query in queries:
        if len(ordered) >= MAX_CANDIDATES:
            break
        try:
            hits = _tool_payload(await search.ainvoke(query))
        except Exception:
            continue
        if not isinstance(hits, list):
            continue
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            person_id = hit.get("id")
            if person_id is None or person_id in seen:
                continue
            seen.add(person_id)
            ordered.append(hit)

    best_profile: dict[str, Any] | None = None
    best_score = -1.0
    for hit in ordered[:MAX_CANDIDATES]:
        try:
            profile = _tool_payload(await lookup.ainvoke({"person_id": hit["id"]}))
        except Exception:
            continue
        if not isinstance(profile, dict) or profile.get("error"):
            continue
        current = _signal_score(claims, profile)
        if current > best_score:
            best_profile, best_score = profile, current
        if best_score >= CONFIDENT_IDENTITY_SCORE:
            break

    if best_profile is None:
        return None
    return {"profile": best_profile, "score": best_score, "pool": len(ordered)}


def _render_claims(claims: dict[str, Any]) -> str:
    """Compact, LLM-friendly rendering of the deterministic CV parse."""
    lines = [
        f"name: {claims.get('name') or '?'}",
        f"headline: {claims.get('headline') or '?'}",
        f"city: {claims.get('city') or '?'}, {claims.get('country') or '?'}",
        "jobs:",
    ]
    for job in claims.get("jobs", []):
        lines.append(
            f"  - {job.get('title')} @ {job.get('company')} "
            f"({job.get('start')} -> {job.get('end') or 'present'})"
        )
    lines.append("education:")
    for entry in claims.get("education", []):
        lines.append(
            f"  - {entry.get('degree')} in {entry.get('field')} "
            f"@ {entry.get('school')} ({entry.get('year')})"
        )
    lines.append(f"skills: {', '.join(claims.get('skills', [])) or '?'}")
    return "\n".join(lines)


def _build_user_message(claims: dict[str, Any], resolved: dict[str, Any] | None) -> str:
    """Assemble the per-CV instruction handed to the agent."""
    parts = [
        "Verify the following CV against the candidate's real social profile.",
        "",
        "=== CV TEXT (as extracted from the PDF) ===",
        claims.get("_raw_text", ""),
        "",
        "=== PARSED CLAIMS ===",
        _render_claims(claims),
    ]
    if resolved:
        profile = resolved.get("profile") or {}
        parts += [
            "",
            "=== PROPOSED IDENTITY MATCH (verify it) ===",
            f"get_linkedin_profile(person_id={profile.get('id')})",
            f"profile name: {profile.get('name')}",
            f"profile headline / city: {profile.get('headline')} / "
            f"{profile.get('city')}, {profile.get('country')}",
            f"identity signal score: {resolved.get('score'):.1f} "
            f"(candidates inspected: {resolved.get('pool')})",
        ]
    else:
        parts += [
            "",
            "=== PROPOSED IDENTITY MATCH ===",
            "automatic lookup failed - search for the candidate yourself with",
            "search_linkedin_people(name, location, industry) and confirm the",
            "right person before comparing.",
        ]
    parts += [
        "",
        "Check every allowed field, apply the tolerance rules, and finish with",
        'the single JSON line {"score": ..., "reasons": [...]}.',
    ]
    return "\n".join(parts)


def _extract_score(result: Any) -> float:
    """Pull the JSON score out of the agent's final assistant message."""
    text = ""
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block)
                for block in content
            )
        if content and str(content).strip():
            text = str(content)
            break

    for pattern in (
        r'\{[^{}]*?"score"\s*:\s*(-?[0-9]*\.?[0-9]+)[^{}]*?\}',
        r"score\s*[:=]\s*(-?[0-9]*\.?[0-9]+)",
        r"(-?[0-9]*\.?[0-9]+)",
    ):
        found = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if found:
            try:
                return _clamp(float(found[-1]))
            except (TypeError, ValueError):
                continue
    raise ValueError("the agent reply contained no usable score")


async def _invoke_agent_with_retry(graph: Any, message: str) -> float:
    """Run the agent on one CV, retrying transient failures with back-off.

    The MCP server is shared with the rest of the class and the model API can
    rate-limit, so a single hiccup must not turn a valid CV into a wrong answer.
    """
    last_error: Exception | None = None
    for attempt in range(AGENT_ATTEMPTS):
        try:
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"recursion_limit": 60},
            )
            return _extract_score(result)
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last_error = exc
            if attempt + 1 < AGENT_ATTEMPTS:
                await asyncio.sleep(1.5 * (attempt + 1))
    raise last_error if last_error is not None else RuntimeError("agent failed")


def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once.

    ``tools`` are the six SocialGraph MCP tools (Facebook + LinkedIn search and
    profile lookup), already wrapped as LangChain tools. You may add your own
    local tools as well.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env.
    """
    ### YOUR CODE HERE
    from langchain.agents import create_agent
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(model=MODEL_NAME, temperature=0.0)
    graph = create_agent(model, tools, system_prompt=VERIFIER_SYSTEM_PROMPT)
    # The runner hands this object straight to score_cvs(). Returning the tool
    # map next to the agent lets the deterministic identity step reuse the same
    # MCP tools without rebuilding the connection.
    return {
        "agent": graph,
        "tools": {getattr(tool, "name", ""): tool for tool in tools},
    }


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV.

    ``cvs`` maps each file name to its text, e.g. ``{"CV_1.pdf": "...", ...}``.
    Return a float in [0, 1] for every file name: higher means the CV is more
    likely consistent with the candidate's LinkedIn/Facebook data. A score
    above 0.5 counts as "valid", 0.5 or below counts as "has discrepancy".

        {"CV_1.pdf": 0.9, "CV_4.pdf": 0.1, ...}

    Catch errors per CV (e.g. a failed API call) and still return a score for
    it: an exception here means no results.csv, which scores zero.

    MCP tools are async, so call your agent with ``await agent.ainvoke(...)``.
    You may verify CVs in parallel (e.g. ``asyncio.gather``), but keep at most
    about 3 CVs in flight (e.g. with ``asyncio.Semaphore(3)``): the MCP server is
    shared by the whole class.
    """
    ### YOUR CODE HERE
    bundle = agent if isinstance(agent, dict) else {}
    graph = bundle.get("agent", agent)
    tools = bundle.get("tools", {})
    if graph is None:
        return {name: FALLBACK_SCORE for name in cvs}

    # The MCP server is shared by the whole class: at most three CVs in flight.
    semaphore = asyncio.Semaphore(3)

    async def verify_one(file_name: str, text: str) -> tuple[str, float]:
        async with semaphore:
            try:
                claims = _cv_claims(text)
                claims["_raw_text"] = text
                try:
                    resolved = await _resolve_person(tools, claims)
                except Exception:
                    resolved = None
                message = _build_user_message(claims, resolved)
                return file_name, await _invoke_agent_with_retry(graph, message)
            except Exception:
                # One bad CV (tool failure, API error, unparseable reply) must
                # never take down the whole run: always return a usable score.
                return file_name, FALLBACK_SCORE

    outcomes = await asyncio.gather(
        *(verify_one(name, text) for name, text in cvs.items())
    )
    return {name: score for name, score in outcomes}


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
