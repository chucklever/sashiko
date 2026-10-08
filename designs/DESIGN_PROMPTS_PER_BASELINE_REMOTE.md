# DESIGN: Review Prompts Selected by Baseline Remote

## Context

A mailing list rarely carries traffic for exactly one code base. netdev
carries iproute2 patches next to kernel networking patches, linux-nfs
carries nfs-utils and libtirpc, and so on. Sashiko's baseline search
already finds the tree a patch applies to: the review repository can
hold any number of remotes, including ones with unrelated history, and
`git.custom_remotes` in `Settings.toml` adds their branches to the
candidate list. A patch for iproute2 posted to netdev fails to apply to
every kernel candidate and succeeds on the iproute2 remote.

What is missing is the last step. The daemon spawns the review worker with
`--baseline <sha>` and `--project`, never `--prompts`, so the worker falls
through `resolve_prompts_path` in `src/main.rs` to the bundled directory
for the project (`kernel/` for `linux`). The iproute2 patch is then
reviewed against the kernel locking and callstack guides and the kernel
maintainer notes, which is wrong in content. An operator who maintains
a prompt tree for the user-space project has no way to name it.

The relevant pieces today:

- `BaselineRegistry::resolve_candidates` (`src/baseline.rs`) builds the
  ordered candidate list. Custom remotes come after the MAINTAINERS
  heuristic and before linux-next, and each candidate is a
  `BaselineResolution::RemoteTarget { url, name, branch }`.
- `Reviewer::prepare_baseline_worktree` (`src/reviewer.rs`) tries
  candidates in order and stops at the first that applies. At that point
  it holds the winning `BaselineResolution`, writes its `as_str()` form
  (`<remote>/<branch>`) to the `baselines` table, logs it, and returns
  only the row id, the worktree, and the review SHA. The remote name does
  not survive to the spawn.
- `run_review_tool_with_cmd` (`src/reviewer.rs`) builds the worker
  command line. The worker already accepts `--prompts <dir>`, and the
  review stages, `@include` resolution, and the `read_prompt` tool all
  resolve against that directory.
- `CustomRemoteSettings` (`src/settings.rs`) is the per-remote config
  record: `name`, `url`, `check_all_branches`, `only_branches`.

## Design Decisions

| Decision | Choice |
|---|---|
| Match key | The remote's configured `name`, never the URL |
| Where the mapping lives | A new optional `prompts` field on `[[git.custom_remotes]]` |
| Granularity | Per remote, not per branch |
| Fallback | No `prompts` on the winning remote, or a win on a non-custom candidate: the project's bundled directory, exactly as today |
| Workflow | Unchanged. The `linux` workflow runs against whatever prompt directory it is given |
| Local review | `--baseline <remote>/<branch>` consults the same mapping when `--prompts` is absent |
| Validation | Each configured `prompts` directory is checked at settings load for the files the workflow asks for by name |

### Why the remote name

The remote name is what the baseline search produces, what the operator
wrote in `Settings.toml`, and what appears in the `baselines` table and
the review log. Matching on it keeps one vocabulary across config, logs,
and code. A URL would be stable across daemons, but the remote is already
defined by name in the same config block, so a second key buys nothing
and invites a mismatch between the two.

### Why per remote rather than per branch

A remote is a code base. Its branches (`main`, `next`, a release branch)
share a language, a coding style, and a set of review guides. Nothing in
a prompt tree varies by branch. If a branch-level override is
ever needed, `only_branches` already enumerates branches per remote, and a
second remote entry pointing at the same URL with a different name and
`prompts` gives branch-level selection without new syntax.

### Why not a new project kind

`ProjectId` selects a workflow and a bundled prompt directory together.
An iproute2 patch should run the kernel-style patch review workflow with
different guides, not a different workflow, and the daemon is started
once per mailing list, not once per code base. A project kind per
user-space tree would multiply daemons and duplicate the workflow for no
gain. The prompt directory is the only thing that changes, so the prompt
directory is the only thing the mapping controls.

## Configuration

```toml
[[git.custom_remotes]]
name = "iproute2"
url = "https://git.kernel.org/pub/scm/network/iproute2/iproute2-next.git"
check_all_branches = false
only_branches = ["main"]
prompts = "/srv/sashiko/prompts/iproute"
```

`prompts` is optional. A relative path is resolved against the daemon's
working directory at settings load and stored absolute, because the
worker is spawned with a cleared environment and must not depend on
where it was started.

Sashiko ships no prompt tree for a user-space project; the operator
supplies one. `sashiko init --prompts` installs the bundle, and a copy
of its `kernel/` directory with the guides rewritten for the project is
the expected starting point, since the `linux` workflow is what reads
the tree.

## Data Flow

1. `resolve_candidates` is unchanged. Custom-remote candidates already
   carry `name`.
2. `prepare_baseline_worktree` returns the winning `BaselineResolution`
   alongside the baseline row id, worktree, and SHA. Its two callers in
   `Reviewer::review_patchset` take the extra field.
3. `Reviewer` looks up the winning candidate: for
   `RemoteTarget { name, .. }`, find the `CustomRemoteSettings` with that
   `name` and take its `prompts`. For `Commit` and `LocalRef`, or a
   `RemoteTarget` whose name is a MAINTAINERS or linux-next remote, the
   result is `None`.
4. `run_review_tool_with_cmd` appends `--prompts <path>` when the lookup
   yields a path. The worker needs no change: `resolve_prompts_path`
   already returns an explicit path verbatim.
5. The chosen path is logged next to the existing "Baseline found for
   patchset" line so the review log answers which guides were used.

For `sashiko review` with `--baseline` and no `--prompts`, step 3 runs on
the given ref: if it has the form `<name>/<branch>` and `<name>` is a
custom remote with `prompts`, that directory is used. Any other ref
falls through to the bundled directory. `--prompts` still wins when
given.

## Validation

The `linux` workflow asks for prompt files by fixed name, as the
`projects/linux/prompts/README.md` layout table lists them:
`severity.md`, `false-positive-guide.md`, `inline-template.md`,
`callstack.md`, `locking.md`, and `pointer-guards.md`. An `@include` of
a missing file resolves to nothing, which is right for an optional
guide and wrong for a tree that is missing the core: a review with no
`severity.md` runs with an empty severity guide and reports nothing
wrong.

Settings validation therefore checks each configured `prompts` directory
for `severity.md`, `false-positive-guide.md`, `inline-template.md`,
`callstack.md`, and `locking.md`, and refuses to start the daemon (or
the local review) naming the directory and the missing file. The set of
required names lives in one place in `src/workflows/linux_patch_review.rs`
beside the stage definitions that use them, so adding a stage that needs
a new file updates the check.

Maintainer notes (`maintainers/*.md`) and code notes (`notes/`) stay
optional. A user-space tree with neither gets no maintainer note
selected and an empty code-notes agenda, which is the intended result.

## Candidate Ordering

Custom remotes are tried before linux-next and mainline. A kernel patch
therefore gets attempted against the iproute2 remote first, fails, and
moves on. The cost is one failed `git apply` per configured user-space
branch per kernel patch. With `only_branches` naming a single branch per
remote this is negligible. Operators should not set `check_all_branches`
on a user-space remote carried on a kernel list; that is already the
guidance for kernel remotes and it holds here for the same reason.

The reverse case is the point of the change: a user-space patch fails on
every kernel candidate (base-commit, version tag, MAINTAINERS heuristic)
and applies to the custom remote. The MAINTAINERS heuristic runs on
paths that mean nothing in the kernel tree and yields no candidates,
which is harmless.

## Recorded Provenance

`reviews.prompts_hash` currently records the daemon's build hash as a
stand-in for "which prompts". With an operator-supplied directory the
build hash no longer describes the prompts. The `baselines` row already
names the winning remote and branch, so the mapping from a review to its
prompt directory is recoverable from the settings in force at the time.
Recording the resolved path on the review row is a follow-up if that
proves insufficient; it is not needed for correctness.

## Out of Scope

- A per-branch mapping syntax. See "Why per remote".
- Choosing a different workflow per remote. The workflow stays `linux`.
- Prompts for a code base that has no remote in the review repository.
  Baseline detection is the trigger; a tree that cannot be found cannot
  be reviewed regardless of prompts.
- Reading `severity.md` from the prompt directory in the bug workflow.
  `prompt_bundle::severity_guide` is compiled in and used only by
  `linux_bug.rs`, which this change does not touch.

## Implementation Plan

Three commits, each leaving `make check-pr` and the local self-review
clean:

1. **settings: Add a prompts directory to custom remotes.** New optional
   field on `CustomRemoteSettings`, path made absolute at load, and the
   required-file validation with its list exported from
   `linux_patch_review.rs`. Unit tests for the validator with a temp
   directory missing each required file in turn.
2. **reviewer: Return the winning baseline from the worktree search.**
   Widen the `prepare_baseline_worktree` result to carry the
   `BaselineResolution`; adjust the two callers. No behavior change.
3. **reviewer: Pass the remote's prompt directory to the worker.** The
   lookup helper on the custom-remote list, the `--prompts` argument in
   `run_review_tool_with_cmd`, the same lookup in
   `handle_review_command` for an explicit `--baseline`, and the log
   line. Tests: a spawn with a matching remote carries `--prompts`; a
   spawn with a mainline win does not; local review resolves
   `iproute2/main` to the configured directory and `v6.12` to the bundle.
