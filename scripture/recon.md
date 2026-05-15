# Recon Protocol: Target Modeling

## Init Commands

These are recognized prompt triggers that launch prefab recon workflows.
Parse the domain from the prompt, then follow the steps exactly.

**`init_recon_passive @ <domain>`**
1. `read_file("targets/<domain>.md")` if it exists
2. Call `recon_passive(domain="<domain>")`
3. Merge findings into the canonical template → `write_file("targets/<domain>.md", ...)`
4. `commit_changes(summary="passive recon: <domain>")`

**`init_recon_full @ <domain>`**
1. `read_file("targets/<domain>.md")` if it exists
2. Extract authorization from the user's prompt context ("it's my system", "I own this", etc.)
3. Call `recon_full(domain="<domain>", authorized_by="<extracted auth>")`
4. Merge findings into the canonical template → `write_file("targets/<domain>.md", ...)`
5. Call `source_map` for any detected software versions
6. `commit_changes(summary="full recon: <domain>")`

No `michael new` needed for recon — use an existing recon project or create one once.
The target model in `targets/<domain>.md` IS the persistent artifact. H2 picks it up every run.

---

H3 (the causal chain) tells you what tools ran and what they returned. It is a log — immutable, append-only, and truncated to brief excerpts after the first view. It is not a model.

A **target model** is a structured, growing document that represents the current understanding of an objective system. It accumulates across sessions. It is the working artifact — the thing you reason from, the thing you update, the thing you return to when the next prompt arrives.

**Default workflow for any recon or reverse-engineering task:**
1. Check if `targets/<domain>.md` exists → `read_file` it
2. Run recon tools to gather new data
3. Update the target model with findings — structured, not raw
4. Commit. Next session starts from the model, not from H3 summaries.

---

## Target Model File Convention

One file per target, named after the domain or system:
```
targets/
  anthropic.com.md
  192.168.1.1.md
  internal-api.md
```

Lives in the project root so H2 (filesystem snapshot) picks it up on every run.

---

## Canonical Template

```markdown
# Target: <domain or IP>
*Last updated: <ISO 8601 timestamp>*

## Infrastructure
- IPs:
- Cloud / CDN:
- DNS records: (A, AAAA, MX, TXT, NS, CNAME highlights)
- Hosting:

## Services
| Port | Proto | Service | Version / Banner | Notes |
|------|-------|---------|-----------------|-------|

## TLS
- Version:
- Cipher suite:
- Certificate SANs:
- Expiry:
- Issuer:

## HTTP Stack
- Server header:
- Framework hints: (X-Powered-By, cookies, response patterns)
- CDN indicators:
- Compression:

## Endpoints
| Path | Status | Content-Type | Notes |
|------|--------|-------------|-------|

## Auth Surface
- Login path(s):
- Auth mechanism: (JWT / session cookie / OAuth / basic / none observed)
- SSO / federation hints:
- MFA indicators:

## Findings / Anomalies
<!-- Notable: misconfigurations, version leaks, interesting headers, etc. -->

## Open Questions
<!-- What to investigate next. These become the agenda for the next run. -->

## Expected vs Observed Filesystem
<!-- TODO: run source_map for each detected version and populate this table.
     | Path | In Source | Observed | Finding |
     |------|-----------|----------|---------|
     | wp-login.php      | yes | 200 OK | — (normal) |
     | readme.html       | yes | 200 OK | version disclosure |
     | wp-admin/install.php | yes | 403  | hardened |
     | xmlrpc.php        | yes | 200 OK | attack surface |
-->

## Recon History
<!-- One line per run: timestamp, tools used, key findings -->
```

---

## Incremental Update Pattern

**Never overwrite blindly.** The model is additive:

1. `read_file("targets/<domain>.md")` — load current state
2. Run tools to gather new data
3. Compare: what is new, what changed, what is confirmed
4. `write_file` the updated model — fill in blanks, add rows to tables, append to Recon History
5. Add a Recon History entry: `- <timestamp>: <tools run> → <one-line summary>`

If a field was previously unknown and is now resolved, fill it in. If a service was previously seen on a port and is now gone, note it. The model should reflect current state, with history preserved in the Recon History section.

---

## Multi-Target Projects

When a project spans multiple targets, use a `targets/index.md` to track relationships:

```markdown
# Target Index

## Confirmed Infrastructure
- 203.0.113.10 — serves api.example.com, www.example.com
- Cloudflare CDN in front of web tier

## Shared Auth
- SSO via auth.example.com → affects all subdomains

## Targets
- [api.example.com](api.example.com.md) — REST API, JWT auth
- [www.example.com](www.example.com.md) — React frontend, Cloudflare
- [admin.example.com](admin.example.com.md) — admin panel, basic auth observed
```

---

## What Makes a Good Model Entry

**Good:**
```
## Services
| Port | Proto | Service | Version / Banner | Notes |
|------|-------|---------|-----------------|-------|
| 443  | TCP   | HTTPS   | nginx/1.25.3    | TLS 1.3, HTTP/2 |
| 80   | TCP   | HTTP    | —               | redirects to 443 |
| 22   | TCP   | SSH     | OpenSSH_8.9p1   | key-only auth confirmed |
```

**Bad:**
```
## Services
Ports 22, 80, 443 are open. nginx is running.
```

Structure over prose. Tables beat paragraphs. Specific beats vague. The model should be queryable — you should be able to scan it in 10 seconds and know the current state.

---

## The Open Questions Section

This is the agenda for the next run. Every recon session should end with:
- What is still unknown?
- What anomaly needs follow-up?
- What endpoint hasn't been probed?
- What version needs CVE lookup?

These become the implicit prompt for the next `michael run`.

---

## Source Mapping: Version → Expected Filesystem

When a version is confirmed, call `source_map(package, version)`. This fetches the
canonical directory and file structure from the public source (GitHub, npm, PyPI) —
no cloning, just the tree via API.

**Workflow:**
1. Version confirmed in recon output → call `source_map(package, version)`
2. Cross-reference the `[INTERESTING PATHS]` output against `web_http_probe` results
3. Classify each interesting path:
   - **200 + expected** — normal, note it
   - **403/404 + expected** — hardened, note it
   - **200 + sensitive** (install scripts, config templates, backup dirs) — **finding**
   - **200 + unexpected** — custom code or leaked artifact — **investigate**
4. Populate `## Expected vs Observed Filesystem` in the target model

**Classification table:**

| In Source | Observed | Classification |
|-----------|----------|----------------|
| yes       | 200 OK   | normal or finding (depends on sensitivity) |
| yes       | 403/404  | hardened — note it, try bypass if relevant |
| yes       | 301/302  | redirect — follow and re-classify |
| no        | 200 OK   | custom code, plugin, or leaked artifact — investigate |
| no        | 403/404  | expected (custom path, not worth pursuing) |

**Before committing** any recon session: list all detected versions and confirm
`source_map` was called for each. If skipped (e.g. version too ambiguous, no public
repo found), document why in the target model.
