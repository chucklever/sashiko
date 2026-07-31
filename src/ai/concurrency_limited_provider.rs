// Copyright 2026 The Sashiko Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A pass-through [`AiProvider`] decorator that caps the number of concurrent
//! `generate_content` calls against a shared semaphore.
//!
//! A review fans its analysis stages out concurrently, so a single patch can
//! issue one model call per stage at once, and the worker reviews several
//! patches in parallel on top of that. The daemon bounds this globally with its
//! own LLM semaphore, but a worker running reviews in-process has nothing in
//! front of it. Sharing one semaphore across every provider the run creates
//! turns `[review] concurrency` into a real ceiling on in-flight model calls.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::Semaphore;

use crate::ai::backoff_provider::RetryBudget;
use crate::ai::{
    AiProvider, AiRequest, AiResponse, CacheStats, LLM_SLOTS_PER_REVIEW, ProviderCapabilities,
};

/// How many permits the LLM semaphore holds for a review `concurrency`.
///
/// Processes and worktrees are gated at `concurrency` itself; a review has
/// more LLM requests than that in flight over its life, which is what
/// `LLM_SLOTS_PER_REVIEW` accounts for.  A call draws
/// `AiProvider::llm_permits_for()` of the result, not always one: a provider
/// that occupies a resource on this host for the call's whole duration weighs
/// `LOCAL_PERMITS_PER_CALL`, so a serial configuration admits exactly one such
/// call while API-backed providers still run a few requests wide.
pub fn llm_permits(concurrency: usize) -> usize {
    std::cmp::max(1, concurrency) * LLM_SLOTS_PER_REVIEW as usize
}

/// Limits concurrent model calls to the permits of a shared semaphore. All
/// other behaviour is delegated unchanged to the inner provider.
pub struct ConcurrencyLimitedProvider {
    inner: Arc<dyn AiProvider>,
    semaphore: Arc<Semaphore>,
    /// Credited with the time a call spends queued for permits.  A task
    /// waiting behind another review emits no output while the review's
    /// deadline runs, so that wait is given back the same way the quota wait
    /// is.
    budget: Option<Arc<dyn RetryBudget>>,
}

impl ConcurrencyLimitedProvider {
    /// The semaphore is shared, so every provider built from it draws on the
    /// same pool of permits.
    pub fn new(
        inner: Arc<dyn AiProvider>,
        semaphore: Arc<Semaphore>,
        budget: Option<Arc<dyn RetryBudget>>,
    ) -> Self {
        Self {
            inner,
            semaphore,
            budget,
        }
    }
}

#[async_trait]
impl AiProvider for ConcurrencyLimitedProvider {
    async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
        let queued_at = tokio::time::Instant::now();
        let _permit = self
            .semaphore
            .acquire_many(self.inner.llm_permits_for(&request).await)
            .await
            .map_err(|e| anyhow::anyhow!("concurrency semaphore closed: {e}"))?;
        if let Some(budget) = &self.budget {
            budget.credit_wait(queued_at.elapsed());
        }
        self.inner.generate_content(request).await
    }

    fn llm_permits(&self) -> u32 {
        self.inner.llm_permits()
    }

    async fn llm_permits_for(&self, request: &AiRequest) -> u32 {
        self.inner.llm_permits_for(request).await
    }

    fn estimate_tokens(&self, request: &AiRequest) -> usize {
        self.inner.estimate_tokens(request)
    }

    fn get_capabilities(&self) -> ProviderCapabilities {
        self.inner.get_capabilities()
    }

    fn cache_stats(&self) -> Option<CacheStats> {
        self.inner.cache_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_permits_scales_with_concurrency() {
        // A serial configuration gets one review's slots: exactly one
        // LOCAL_PERMITS_PER_CALL call, or a few API-backed requests.
        assert_eq!(llm_permits(0), LLM_SLOTS_PER_REVIEW as usize);
        assert_eq!(llm_permits(1), LLM_SLOTS_PER_REVIEW as usize);

        // Above that, calls are allowed to run wider than the worktrees are.
        assert_eq!(llm_permits(2), 2 * LLM_SLOTS_PER_REVIEW as usize);
        assert_eq!(llm_permits(16), 16 * LLM_SLOTS_PER_REVIEW as usize);
    }
}
