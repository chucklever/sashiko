use crate::settings::SmtpSettings;
use lettre::message::header::ContentType;
use lettre::transport::smtp::authentication::Credentials;
use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{error, info, warn};

pub struct EmailWorker {
    db: std::sync::Arc<crate::db::Database>,
    settings: SmtpSettings,
}

impl EmailWorker {
    pub fn new(db: std::sync::Arc<crate::db::Database>, settings: SmtpSettings) -> Self {
        Self { db, settings }
    }

    pub async fn run(&self) {
        info!("Starting Email Worker...");
        loop {
            // Reclaim ghost emails (crashed while sending)
            if let Err(e) = self.db.sweep_ghost_emails().await {
                error!("Failed to sweep ghost emails: {}", e);
            }

            // Lock and send next pending email
            match self.db.lock_pending_email().await {
                Ok(Some(email)) => {
                    info!(
                        "Locked pending email ID {} for patch {:?}",
                        email.id, email.patch_id
                    );
                    match self.send_email(&email).await {
                        Ok(_) => {
                            info!("Successfully sent email ID {}", email.id);
                            if let Err(e) = self.db.mark_email_sent(email.id).await {
                                error!("Failed to mark email {} as sent: {}", email.id, e);
                            }
                        }
                        Err(e) => {
                            error!("Failed to send email ID {}: {}", email.id, e);
                            if let Err(db_err) =
                                self.db.mark_email_failed(email.id, &e.to_string()).await
                            {
                                error!("Failed to mark email {} as failed: {}", email.id, db_err);
                            }
                        }
                    }
                }
                Ok(None) => {
                    // No pending emails, sleep
                    sleep(Duration::from_secs(5)).await;
                }
                Err(e) => {
                    error!("Database error while locking pending email: {}", e);
                    sleep(Duration::from_secs(10)).await;
                }
            }
        }
    }

    fn build_message(&self, email_row: &crate::db::EmailOutboxRow) -> anyhow::Result<Message> {
        let mut builder = Message::builder()
            .from(self.settings.sender_address.parse()?)
            .subject(&email_row.subject);

        if let Some(reply_to) = &self.settings.reply_to {
            match reply_to.parse() {
                Ok(addr) => builder = builder.reply_to(addr),
                Err(e) => warn!("Failed to parse reply_to address '{}': {}", reply_to, e),
            }
        }

        let to_addresses: Vec<String> = serde_json::from_str(&email_row.to_addresses)?;
        for to in to_addresses {
            match parse_lenient(&to) {
                Ok(addr) => builder = builder.to(addr),
                Err(e) => warn!("Failed to parse 'to' address '{}': {}", to, e),
            }
        }

        let cc_addresses: Vec<String> = serde_json::from_str(&email_row.cc_addresses)?;
        for cc in cc_addresses {
            match parse_lenient(&cc) {
                Ok(addr) => builder = builder.cc(addr),
                Err(e) => warn!("Failed to parse 'cc' address '{}': {}", cc, e),
            }
        }

        if !email_row.in_reply_to.is_empty() {
            builder = builder.header(lettre::message::header::InReplyTo::from(format!(
                "<{}>",
                email_row.in_reply_to
            )));
        }

        if !email_row.references_hdr.is_empty() {
            let refs: Vec<String> = email_row
                .references_hdr
                .split_whitespace()
                .map(|part| format!("<{}>", part))
                .collect();
            builder = builder.references(refs.join(" "));
        }

        Ok(builder
            .header(ContentType::TEXT_PLAIN)
            .body(email_row.body.clone())?)
    }

    async fn send_email(&self, email_row: &crate::db::EmailOutboxRow) -> anyhow::Result<()> {
        if self.settings.dry_run {
            info!(
                "DRY RUN: Would have sent email to {}, cc {}, subject '{}'",
                email_row.to_addresses, email_row.cc_addresses, email_row.subject
            );
            info!("DRY RUN Body:\n{}", email_row.body);
            return Ok(());
        }

        let msg = self.build_message(email_row)?;

        let mut mailer_builder =
            AsyncSmtpTransport::<Tokio1Executor>::relay(&self.settings.server)?
                .port(self.settings.port);

        if let (Some(user), Some(pass)) = (&self.settings.username, &self.settings.password) {
            let creds = Credentials::new(user.to_string(), pass.to_string());
            mailer_builder = mailer_builder.credentials(creds);
        }

        let mailer = mailer_builder.build();

        mailer.send(msg).await?;

        Ok(())
    }
}

fn parse_lenient(s: &str) -> anyhow::Result<lettre::message::Mailbox> {
    if let Some(start) = s.find('<')
        && let Some(end) = s.rfind('>')
        && start < end
    {
        let name = s[..start].trim();
        let email = s[start + 1..end].trim();
        let addr: lettre::Address = email.parse()?;
        if name.is_empty() {
            return Ok(lettre::message::Mailbox::new(None, addr));
        } else {
            let clean_name = name.trim_matches('"').to_string();
            return Ok(lettre::message::Mailbox::new(Some(clean_name), addr));
        }
    }
    let addr: lettre::Address = s.parse()?;
    Ok(lettre::message::Mailbox::new(None, addr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_email_parsing() {
        let addr_str = "\"Thomas Richard (TI)\" <thomas.richard@bootlin.com>";
        let parsed = parse_lenient(addr_str);
        assert!(parsed.is_ok(), "Failed to parse: {:?}", parsed.err());
        assert_eq!(
            format!("{}", parsed.unwrap()),
            "\"Thomas Richard (TI)\" <thomas.richard@bootlin.com>"
        );
    }

    #[test]
    fn test_email_parsing_unquoted() {
        let addr_str = "Thomas Richard (TI) <thomas.richard@bootlin.com>";
        let parsed = parse_lenient(addr_str);
        assert!(parsed.is_ok(), "Failed to parse: {:?}", parsed.err());
        assert_eq!(
            format!("{}", parsed.unwrap()),
            "\"Thomas Richard (TI)\" <thomas.richard@bootlin.com>"
        );
    }

    #[test]
    fn test_email_parsing_plain() {
        let addr_str = "thomas.richard@bootlin.com";
        let parsed = parse_lenient(addr_str);
        assert!(parsed.is_ok(), "Failed to parse: {:?}", parsed.err());
        // We will see what format!() returns for plain email
        info!("Plain email formatted: {}", parsed.as_ref().unwrap());
    }
}
