use crate::git_ops::{FetchOutcome, ensure_remote};
use anyhow::Result;
use std::path::PathBuf;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::sleep;
use tracing::{error, info, warn};

pub struct GitSyncWorker {
    repo_path: PathBuf,
}

impl GitSyncWorker {
    pub fn new(repo_path: PathBuf) -> Self {
        Self { repo_path }
    }

    pub async fn run(&self) {
        info!("GitSyncWorker started. Will sync remotes periodically.");
        loop {
            if let Err(e) = self.sync_all_remotes().await {
                error!("GitSyncWorker failed during sync cycle: {}", e);
            }
            // Sleep for 1 hour before checking again.
            // ensure_remote handles the fine-grained 4h/24h timestamp logic.
            sleep(Duration::from_secs(3600)).await;
        }
    }

    async fn sync_all_remotes(&self) -> Result<()> {
        info!("GitSyncWorker: Starting sync cycle.");

        // Enumerate all configured remotes
        let output = Command::new("git")
            .current_dir(&self.repo_path)
            .args(["remote"])
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("GitSyncWorker: Failed to list git remotes: {}", stderr);
            return Err(anyhow::anyhow!("Failed to list remotes"));
        }

        let remotes_str = String::from_utf8_lossy(&output.stdout);
        let remotes: Vec<&str> = remotes_str
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with("fetcher-"))
            .collect();

        info!("GitSyncWorker: Found {} remotes to check.", remotes.len());

        let mut fetched = 0usize;
        let mut skipped = 0usize;
        let mut stale = 0usize;
        let mut failed = 0usize;
        let mut first_failure = String::new();

        for remote in remotes {
            // Get URL for the remote
            let url_output = Command::new("git")
                .current_dir(&self.repo_path)
                .args(["remote", "get-url", remote])
                .output()
                .await?;

            if !url_output.status.success() {
                let stderr = String::from_utf8_lossy(&url_output.stderr);
                let message = format!("Failed to get URL for remote {}: {}", remote, stderr.trim());
                warn!("GitSyncWorker: {}", message);
                failed += 1;
                if first_failure.is_empty() {
                    first_failure = message;
                }
                continue;
            }

            let url = String::from_utf8_lossy(&url_output.stdout)
                .trim()
                .to_string();

            // Check if it's time to fetch and fetch if necessary
            // force_fetch=false so we respect the 4h/24h intervals in ensure_remote
            match ensure_remote(&self.repo_path, remote, &url, false).await {
                Ok(FetchOutcome::Fetched) => fetched += 1,
                Ok(FetchOutcome::Skipped) | Ok(FetchOutcome::BackedOff) => skipped += 1,
                Ok(FetchOutcome::StaleLocalRef(message)) => {
                    stale += 1;
                    if first_failure.is_empty() {
                        first_failure = message;
                    }
                }
                Err(e) => {
                    error!("GitSyncWorker: Failed to sync remote {}: {}", remote, e);
                    failed += 1;
                    if first_failure.is_empty() {
                        first_failure = e.to_string();
                    }
                }
            }
        }

        // A cycle that tries several remotes and reaches none is a
        // condition of the repository or the network rather than of
        // any one remote, and the per-remote errors above do not say
        // so anywhere.  One remote tried and lost is that remote's
        // condition, and its own line already names it.
        let tried = fetched + stale + failed;
        if fetched == 0 && tried > 1 {
            error!(
                "GitSyncWorker: Sync cycle fetched nothing: {} skipped, {} stale, {} failed. First failure: {}",
                skipped, stale, failed, first_failure
            );
        } else {
            info!(
                "GitSyncWorker: Sync cycle complete: {} fetched, {} skipped, {} stale, {} failed.",
                fetched, skipped, stale, failed
            );
        }
        Ok(())
    }
}
