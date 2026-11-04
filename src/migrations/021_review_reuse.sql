-- Copyright 2026 The Sashiko Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     https://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- A review the worker did not run because an identical diff had already
-- been reviewed with no findings. The row points at the review whose
-- verdict it inherits; see designs/DESIGN_SKIP_UNCHANGED_CLEAN_PATCHES.md.
ALTER TABLE reviews ADD COLUMN inherited_from_review_id INTEGER
    REFERENCES reviews(id);

-- The commit the review's worktree was checked out at. baselines.
-- last_known_commit moves with every sync, so a review needs its own
-- record of the tree it ran against before two reviews can be compared
-- by baseline.
ALTER TABLE reviews ADD COLUMN baseline_commit TEXT;

-- Foreign key enforcement checks this column whenever a reviews row is
-- deleted; without the index that check scans the table per row.
CREATE INDEX IF NOT EXISTS idx_reviews_inherited_from_review_id
    ON reviews(inherited_from_review_id);
