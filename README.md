# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1

A deterministic stage resolves *who* the candidate is; an agent then judges the
claims. The resolver regex-parses the CV (flattening markdown tables first — one
public CV extracts as a table), takes the industry word from the headline,
searches LinkedIn by name, location and industry, and ranks up to eight
candidates by overlap of employers, schools, skills and city. The name query
alone returns 20 of several hundred namesakes. The judge is one
`deepseek-v4-flash` tool-calling agent whose prompt fixes the five field groups
it may flag, the tolerance rules (Senior X for X with seniority senior, fewer
skills is fine) and a one-year drift as a discrepancy. `score_cvs` keeps three
CVs in flight, retries failures with back-off, and always returns a float.

Public test: **7/7 in 166 s** — CV_1 to CV_7 scored 0.95, 0.97, 0.95, 0.40,
0.45, 0.96, 0.45. An earlier 5/7 came from transient call failures the exception
path swallowed; the retry loop fixed it.

### Task 2

Two graders, two rule books. The verifiers are bound to the Task 1 spec, where
discrepancies live only in name, city, jobs, education and skills; you compare
against the real profile and count anything fabricated. A claim with no slot in
the profile schema falls between them.

The CV is Kelly Tsang's real one with every checkable field untouched, plus one
visible line: `Certifications: CFA Level II Candidate (May 2024)`. She holds no
certification, so it is a genuine fabrication, and it sits outside the five field
groups.

White 8-point text at the top carries an indirect injection — invisible on
screen, ordinary text in the layer the verifier reads. It restates the scope
rule, extends it to cover certifications, adds a document-control cue and
suggests a rating. Credential and injection sit above the first heading so
section-tracking parsers call them header noise; my first attempt put the
credential under Skills and my own parser folded it, plus the hidden block, into
the skill list.

The PDF is built with reportlab: LaTeX output extracts as markdown tables.

Result on my agent: adversarial 0.97, target 0.97.

### Task 3: Reflection

The verifier is the forger's teacher. Building Task 1 showed me where a CV could
lie: the five fields it checks are the five worth attacking, and everything
outside them is a blind spot it is required to have.

A score describes documents, not the world. My agent gave the forged CV 0.97 and
the real one 0.97; both numbers only say two documents agree.

Reading is the attack surface. An agent must interpret untrusted text to compare
a job title, and the same faculty lets a white paragraph tell it what to
conclude. One model family writes the verifier and, with the same tools, the CV
that beats it. Adding certifications to the schema closes my trick and moves the
next one elsewhere; the fix is treating extracted text as data with provenance,
and keeping a human answerable where claims and record disagree.
