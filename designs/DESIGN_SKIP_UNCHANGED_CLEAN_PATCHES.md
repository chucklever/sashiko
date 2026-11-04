# Skipping Unchanged Patches That Already Reviewed Clean

## 1. Problem Statement

Large series are reposted many times. A 30-patch series at v7 has
typically changed in two or three patches since v6; the other
twenty-odd patches carry the same diff, often the same commit message,
and sometimes only a rebased `index` line. Sashiko reviews every patch
of every posting from scratch. When the prior review of an unchanged
patch produced no findings, the new review almost always produces no
findings again, at full token cost.

Today the daemon has no way to avoid that spend:

- A patch is deduplicated only by `message_id`
  (`ON CONFLICT(message_id) DO NOTHING` in `src/db.rs`). A repost is
  a new message, so it is a new patch row and a new review.
- `prior_context.rs` already detects the unchanged case and tells the
  model `(patch diff unchanged)` alongside the prior findings. That
  improves consistency but does not shorten the review: every stage
  still runs with the full series as context.
- The `Skipped` patch status exists, but it is only set for size
  limits (`review.max_lines_changed`, `review.max_files_touched`) and
  for the benchmark runner's `skip_patch_ids` override.

The motivating workload is the kernel mailing lists, where repeated
postings of sparse-change series are routine and the token budget is
finite.

## 2. Goals

- Skip the LLM review of a patch whose normalized diff is identical to
  a patch Sashiko already reviewed to completion with zero findings.
- Keep the skipped patch visible in the series: it must still be
  present in the `patches_json` input of the patches that are
  reviewed, so cross-patch context is unchanged.
- Record why the patch was skipped and which prior review it inherits
  from, so the dashboard can explain the outcome.
- Default off. Operators opt in per deployment.

## 3. Non-Goals

- Reusing a prior review that had findings. A finding needs a fresh
  verification pass against the current baseline and the author's
  response; that is the existing prior-version feedback flow in
  `prior_feedback.rs` and is unaffected here.
- Skipping based on commit message changes alone. Only the diff is
  compared. A reworded commit message on an identical diff still
  skips (the commit message is not where bugs live), and that is the
  intended trade-off.
- Cross-project matching. A daemon database serves one project (the
  `project` key in the `meta` table), so a lookup never meets a patch
  from another project.
- Any change to the local `sashiko review` command. It has no
  database and reviews whatever commits it is given.

## 4. Anti-Goals

- Silently dropping a patch from the series without a record. Every
  skip must leave a `reviews` row a maintainer can find.
- Skipping when the prior review did not finish (`Failed`,
  `Cancelled`) or when it reported pre-existing findings only. "No
  findings" means a `Reviewed` review with zero `findings` rows of any
  kind.

## 5. Design

### 5.1 Identity: exact diff, indexed by Git patch ID

Two patches carry the same change when their diffs are equal line for
line after dropping what a rebase rewrites: the mail text before the
first `diff --git` header, the blob SHAs on `index` lines, and the line
ranges on `@@` hunk headers. The first two are the normalization
`compute_inter_version_diff` in `src/prior_context.rs` already applies;
`patch_diffs_equivalent` exposes it as a predicate and adds the third,
because a hunk that moved only because code above it grew or shrank is
the same hunk. The function context after the ranges still has to
match. Whitespace stays significant. A reindented hunk is a different
patch.

The database needs no new column to find candidates. `patches.
git_patch_id` is computed by the ingest path for every patch and is
already indexed (migration 12). `git patch-id --stable` ignores
whitespace and line numbers, so it is a superset of the identity above:
every exact match shares a patch ID, and a patch ID match is confirmed
by comparing the diffs in Rust. A patch with no stored ID (an empty
diff) is never skipped.

### 5.2 Lookup

`Database::find_clean_review_candidates` runs from `process_patch_review`
in `src/reviewer.rs`, after the ignored-files check and before a review
row is claimed:

```sql
SELECT r.id, p.id, p.patchset_id, ps.subject, ps.version, p.part_index,
       p.diff, r.model, r.prompts_hash, r.baseline_commit
FROM patches p
JOIN reviews r ON r.patch_id = p.id
JOIN patchsets ps ON ps.id = p.patchset_id
WHERE p.git_patch_id = ?1
  AND p.id != ?2
  AND r.status = 'Reviewed'
  AND r.created_at >= ?3
  AND r.inherited_from_review_id IS NULL
ORDER BY r.created_at DESC, r.id DESC
LIMIT 8
```

`?3` is `now - max_age`. Each row also carries whether the review has
any `findings`. The reviewer walks the candidates newest first and
stops at the first whose diff passes `patch_diffs_equivalent`: if that
review found something, nothing older is reused, because the newest
verdict on a diff overrules every clean one before it; if it was clean
and meets the conditions in 5.3, it is the source. The limit keeps a
patch carried across many series from pulling every copy's diff out of
the database.

Reviews that themselves inherited a verdict are excluded, so a chain of
reposts always ends at the review a model produced, and the age bound
is measured from that review, not from the latest repost.

Lineage (`patches.previous_patch_id`) is not consulted. A patch moved
between series, or resent under a new thread, matches the same way as
one in the same version chain. The prior patchset's subject and part
index label the match for humans.

### 5.3 Skip conditions

All of the following must hold:

1. `review.skip_unchanged_clean = true`.
2. The patchset's `target_review_count` is one. `sashiko-cli rerun`
   raises it to force a model run, and the skip must not satisfy that
   request with another inherited row.
3. The patch's `PriorVersionContext` is empty for audit: no
   `prior_findings` and no `discussion_threads`. Either one means the
   prior-feedback stage has work to do on this patch, and that stage
   only runs inside a full review. The check reads the same
   `prior_version_context` the model would be given.
4. A matching prior review exists per 5.2, and it is clean.
5. The prior review's `model` and `prompts_hash` match the current
   worker's, unless `review.skip_unchanged_clean_any_model = true`.
   The default is strict: a prompt or model change is exactly when a
   re-review of an old clean patch is worth paying for.
6. If `review.skip_unchanged_clean_require_same_baseline = true`, the
   current baseline is the same commit as the prior review's
   `baseline_commit`. Off by default.

Condition 6 is relaxed by default on purpose. A series reposted a week
later lands on a newer `-next` tip, so the baseline nearly always
moves, and a strict check would save nothing on the motivating
workload. The same diff on a different tree can be a different patch,
which is why the check exists at all; operators reviewing against a
fast-moving tree where that matters can turn it on. The age bound in
5.6 limits how far the baseline can drift under the default.

### 5.4 What happens on a skip

The skip leaves behind exactly what the existing ignored-files skip
does, plus the link to the inherited review:

- A `reviews` row is claimed through `claim_or_create_in_review` and
  completed with `status = 'Skipped'` and a `result_description` of
  the form `Skipped: unchanged since review <id> of "<subject>" (patch
  <n>) found no issues`. The row carries the current worker's `model`
  and `prompts_hash`, as any review row does, and the new
  `inherited_from_review_id` names the review whose verdict it reuses.
- `inherited_from_review_id` and `baseline_commit` are written while
  the row is still `In Review`, before the status change, so a failure
  between the two leaves a row the orphan cleanup already handles
  rather than a `Skipped` row with no source. Reclaiming a `Pending`
  row for a model run clears both columns again, so a review the model
  produces is never mistaken for an inherited one. Provenance is only
  ever read together with `status = 'Skipped'`.
- The patch gets status `Skipped`.
- A retry of an interrupted series stops at the inherited row the same
  way it stops at a `Reviewed` one: `count_successful_reviews` counts
  a `Skipped` row with provenance as one finished pass. The early
  return then leaves the patch `Skipped` instead of marking it
  `Reviewed`.
- No AI interaction is created and no tokens are counted.
- No outbox row is written. The ignored-files skip writes none either;
  `queue_patchset_notifications` only considers `Reviewed` rows, so a
  skipped patch is silent in mail and contributes nothing to a forge
  comment.
- The patch stays in `patches_json` for the other jobs with its
  `prior_version_context` intact.

Every normal review now also records `reviews.baseline_commit`, the
commit its worktree was checked out at, so the same-baseline rule in
5.3 has something to compare. `baselines.last_known_commit` cannot
serve: it moves with every sync.

### 5.5 Whole-series skip

If every patch in a patchset is skipped, the patchset still completes
as `Reviewed` through the normal path (zero valid jobs is already a
handled state for size-limited series).

Three readers of the result needed to learn the new row. The clean
patchset predicate behind early embargo release,
`get_patchset_review_outcome`, and `count_successful_reviews` all
counted `Reviewed` rows toward a patch's target review count, and the
first two recognized only the ignored-files `Skipped` row as an
exemption from it. A `Skipped` row with `inherited_from_review_id` set
now counts as one finished review in all three. Counting it, rather
than exempting the patch, is what makes a rerun behave: the rerun
raises the target to two, the inherited row supplies one, and the model
runs exactly once more. A reposted series whose every patch inherited
is clean and releasable rather than stranded as incomplete.

### 5.6 Settings

```toml
[review]
# Skip the LLM review of a patch whose diff is byte-identical, modulo
# index lines, to a patch already reviewed with no findings.
skip_unchanged_clean = false
# Only reuse a review produced within this many days.
skip_unchanged_clean_max_age_days = 90
# Reuse a clean review even when the model or prompts have changed.
skip_unchanged_clean_any_model = false
# Reuse a clean review only when the baseline commit is the same.
skip_unchanged_clean_require_same_baseline = false
```

These live on `ReviewSettings` in `src/settings.rs` and are documented
in `docs/configuration.md`. `sashiko review` ignores all four.

### 5.7 Observability

- `info!` on every skip, naming patchset, index, patch id, and the
  inherited review with its patchset and part.
- A `reviews_skipped_unchanged` counter in `src/metrics.rs`, exposed
  by the stats endpoint. It counts since process start; the durable
  record is `inherited_from_review_id`.
- The dashboard shows the review's `result_description` as the
  tooltip of a skipped patch in the list and as the placeholder in the
  patch pane, so the inherited review's id is one hover away.

## 6. Risks

**A clean review was wrong, and the skip propagates the miss.** A
false negative in v3 now also suppresses v4 through v9. Mitigations:
the `max_age` bound, the strict model/prompt default so a prompt fix
re-reviews everything, the newest-verdict rule so one later review
that catches the bug ends the chain, and `sashiko-cli rerun`, which
raises the target review count and thereby disables the skip for that
patchset. This is the accepted cost of the feature and the reason it
is off by default.

**Baseline drift changes the meaning of an identical diff.** Accepted
by default: the age bound caps the drift, and condition 4 offers a
strict mode for trees where it is not acceptable.

**Git patch-id groups patches that differ only in whitespace.** That
is why the patch ID is only the index and the diffs are compared byte
for byte before a verdict is reused.

**Old reviews have no baseline_commit.** The column is filled from the
first review after this change ships. Under the default the column is
not consulted; with the same-baseline rule on, a review with a NULL
baseline is simply never reused.

## 7. Alternatives Considered

**Reuse reviews with findings too, by replaying the prior findings.**
Rejected. Prior findings need re-verification against the author's
changelog and the discussion thread; that is what `prior_feedback.rs`
does and it needs the LLM.

**Match on `previous_patch_id` lineage only.** Simpler, but misses the
common case of a patch moved between series or resent under a new
thread, and lineage reconciliation is itself heuristic. The patch ID
index plus an exact diff comparison is cheaper and more reliable.

**Compare the full patch including the commit message.** Would skip far
less often (authors touch commit messages on nearly every repost) for
no safety gain, since the review targets the code.

**A dedicated normalized-diff hash column.** The first draft of this
design added `patches.diff_hash` with its own index and a lazy backfill.
`git_patch_id` already exists, is indexed, and is computed at ingest for
every patch; one in-process comparison per candidate makes it an exact
identity. The column and the backfill were dropped.

**Rely on the AI response cache.** The cache keys on the full request,
which includes the whole series as context, so a one-patch change in
the series invalidates every entry. It also still runs the workflow
engine and counts cached tokens.

## 8. Implementation Steps

Each step is one commit, in this order on the branch:

1. Migration 19: `reviews.inherited_from_review_id` and
   `reviews.baseline_commit`.
2. `patch_diffs_equivalent` in `prior_context.rs`, with a test for the
   rebase-only case, a whitespace change, and a one-line change.
3. `ReviewSettings` fields, defaults, `Settings.toml` comments,
   `docs/configuration.md`.
4. `Database::find_clean_review_candidates` and the two column
   writers, with a test exercising each predicate.
5. `CLEAN_PATCHSET_PREDICATE` and `get_patchset_review_outcome` accept
   an inherited row.
6. Reviewer: the skip branch and its gate, baseline commit recording,
   metrics, logging. Tests: skip taken end to end, a retry reusing the
   row, each refusal, and the gate.
7. Dashboard: show the skip reason.

## 9. Open Decisions For The Operator

These do not block implementation; the defaults above are the proposed
answers.

- Whether 90 days is the right default age bound. With the baseline
  check relaxed by default, this bound is the only limit on how stale
  a reused clean verdict can be.
