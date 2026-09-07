use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use quack_protocol::{QuackClientOptions, QuackPool, QuackPoolOptions, DEFAULT_MAX_CONNECTIONS};
use secrecy::{ExposeSecret, SecretString};
use snafu::prelude::*;

use crate::conn::QuackConnection;
use datafusion_table_providers_common::sql::db_connection_pool::{
    dbconnection::DbConnection, DbConnectionPool, JoinPushDown,
};

type PoolResult<T, E = Box<dyn std::error::Error + Send + Sync>> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Unable to connect to Quack server '{uri}': {source}"))]
    ConnectionError {
        uri: String,
        source: quack_protocol::QuackError,
    },

    #[snafu(display("Missing required parameter: {param}"))]
    MissingParameter { param: String },

    #[snafu(display("Invalid value for parameter '{param}': {reason}"))]
    InvalidParameter { param: String, reason: String },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Everything needed to open a [`QuackPool`], parsed from the parameter map.
pub struct QuackPoolConfig {
    pub uri: String,
    pub client_options: QuackClientOptions,
    pub pool_options: QuackPoolOptions,
}

impl std::fmt::Debug for QuackPoolConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `QuackClientOptions` derives `Debug` and would print the auth token.
        f.debug_struct("QuackPoolConfig")
            .field("uri", &self.uri)
            .field(
                "auth_token",
                &self
                    .client_options
                    .auth_token
                    .as_ref()
                    .map(|_| "<redacted>"),
            )
            .field("ssl", &self.client_options.ssl)
            .field("timeout", &self.client_options.timeout)
            .field("max_connections", &self.pool_options.max_connections)
            .finish()
    }
}

impl QuackPoolConfig {
    /// Parses the provider's parameter map.
    ///
    /// | key               | meaning                                                           |
    /// |-------------------|-------------------------------------------------------------------|
    /// | `uri`             | required; `host:port`, `quack:host:port` or `http(s)://host:port` |
    /// | `auth_token`      | token sent with every request                                     |
    /// | `max_connections` | sessions the pool may hold open at once (default 4)               |
    /// | `ssl`             | `true`/`false`; overrides what the URI scheme implies             |
    /// | `timeout`         | per-request timeout in seconds                                    |
    pub fn from_params(params: &HashMap<String, SecretString>) -> Result<Self> {
        let get = |key: &str| {
            params
                .get(key)
                .map(|v| v.expose_secret().trim().to_string())
        };

        let uri = get("uri")
            .filter(|uri| !uri.is_empty())
            .context(MissingParameterSnafu { param: "uri" })?;

        let mut client_options = QuackClientOptions {
            auth_token: get("auth_token").filter(|token| !token.is_empty()),
            ..Default::default()
        };

        if let Some(ssl) = get("ssl") {
            client_options.ssl = Some(parse_bool("ssl", &ssl)?);
        }

        if let Some(timeout) = get("timeout") {
            let secs: u64 = timeout.parse().map_err(|_| Error::InvalidParameter {
                param: "timeout".to_string(),
                reason: format!("expected a number of seconds, got '{timeout}'"),
            })?;
            client_options.timeout = Some(Duration::from_secs(secs));
        }

        let max_connections = match get("max_connections") {
            Some(value) => match value.parse::<usize>() {
                Ok(n) if n >= 1 => n,
                _ => {
                    return InvalidParameterSnafu {
                        param: "max_connections",
                        reason: format!("expected an integer >= 1, got '{value}'"),
                    }
                    .fail()
                }
            },
            None => DEFAULT_MAX_CONNECTIONS,
        };

        Ok(Self {
            uri,
            client_options,
            pool_options: QuackPoolOptions { max_connections },
        })
    }
}

fn parse_bool(param: &str, value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        other => InvalidParameterSnafu {
            param,
            reason: format!("expected true or false, got '{other}'"),
        }
        .fail(),
    }
}

/// A [`DbConnectionPool`] over a [`QuackPool`].
///
/// One `QuackPool` is shared by every connection handed out; the Quack pool
/// itself does the per-query session leasing, so `connect()` is free and each
/// concurrent scan ends up on its own server session.
#[derive(Clone)]
pub struct QuackConnectionPool {
    pool: QuackPool,
    join_push_down: JoinPushDown,
}

impl std::fmt::Debug for QuackConnectionPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuackConnectionPool")
            .field("max_connections", &self.pool.max_connections())
            .field("join_push_down", &self.join_push_down)
            .finish()
    }
}

impl QuackConnectionPool {
    /// Opens a pool from a parameter map; see [`QuackPoolConfig::from_params`]
    /// for the accepted keys. The first session is opened eagerly so a bad
    /// URI or token fails here rather than on the first scan.
    pub async fn new(params: HashMap<String, SecretString>) -> Result<Self> {
        Self::from_config(QuackPoolConfig::from_params(&params)?).await
    }

    pub async fn from_config(config: QuackPoolConfig) -> Result<Self> {
        let QuackPoolConfig {
            uri,
            client_options,
            pool_options,
        } = config;

        let pool = QuackPool::connect(&uri, client_options, pool_options)
            .await
            .context(ConnectionSnafu { uri: uri.clone() })?;

        Ok(Self::from_quack_pool(pool, &uri))
    }

    /// Wraps an already-open [`QuackPool`]. `uri` only feeds the join
    /// push-down context: tables from pools with the same URI may be joined
    /// server-side.
    #[must_use]
    pub fn from_quack_pool(pool: QuackPool, uri: &str) -> Self {
        Self {
            pool,
            join_push_down: JoinPushDown::AllowedFor(format!("uri={uri}")),
        }
    }

    /// The underlying Quack pool, for DDL/DML or session-bound work.
    #[must_use]
    pub fn pool(&self) -> QuackPool {
        self.pool.clone()
    }

    /// Closes every session on the server. Dropping the pool does this
    /// best-effort in the background; call this when the outcome matters.
    pub async fn close(&self) -> Result<(), quack_protocol::QuackError> {
        self.pool.close().await
    }
}

#[async_trait]
impl DbConnectionPool<QuackPool, ()> for QuackConnectionPool {
    async fn connect(&self) -> PoolResult<Box<dyn DbConnection<QuackPool, ()>>> {
        Ok(Box::new(QuackConnection::new(self.pool())))
    }

    fn join_push_down(&self) -> JoinPushDown {
        self.join_push_down.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(pairs: &[(&str, &str)]) -> HashMap<String, SecretString> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), SecretString::from((*v).to_string())))
            .collect()
    }

    #[test]
    fn uri_is_required() {
        let err = QuackPoolConfig::from_params(&params(&[])).unwrap_err();
        assert!(matches!(err, Error::MissingParameter { ref param } if param == "uri"));

        let err = QuackPoolConfig::from_params(&params(&[("uri", "  ")])).unwrap_err();
        assert!(matches!(err, Error::MissingParameter { .. }));
    }

    #[test]
    fn defaults_apply_when_only_uri_is_given() {
        let config = QuackPoolConfig::from_params(&params(&[("uri", "quack:localhost:9494")]))
            .expect("valid params");
        assert_eq!(config.uri, "quack:localhost:9494");
        assert_eq!(config.client_options.auth_token, None);
        assert_eq!(config.client_options.ssl, None);
        assert_eq!(config.pool_options.max_connections, DEFAULT_MAX_CONNECTIONS);
        // The client's own defaults (heartbeat, request timeout) are kept.
        assert!(config.client_options.timeout.is_some());
    }

    #[test]
    fn all_parameters_are_parsed() {
        let config = QuackPoolConfig::from_params(&params(&[
            ("uri", "https://quack.example.com:443"),
            ("auth_token", "super_secret"),
            ("max_connections", "8"),
            ("ssl", "true"),
            ("timeout", "30"),
        ]))
        .expect("valid params");
        assert_eq!(
            config.client_options.auth_token.as_deref(),
            Some("super_secret")
        );
        assert_eq!(config.client_options.ssl, Some(true));
        assert_eq!(config.client_options.timeout, Some(Duration::from_secs(30)));
        assert_eq!(config.pool_options.max_connections, 8);
    }

    #[test]
    fn empty_auth_token_means_none() {
        let config =
            QuackPoolConfig::from_params(&params(&[("uri", "localhost:9494"), ("auth_token", "")]))
                .expect("valid params");
        assert_eq!(config.client_options.auth_token, None);
    }

    #[test]
    fn invalid_max_connections_is_rejected() {
        for bad in ["0", "-1", "many"] {
            let err = QuackPoolConfig::from_params(&params(&[
                ("uri", "localhost:9494"),
                ("max_connections", bad),
            ]))
            .unwrap_err();
            assert!(
                matches!(err, Error::InvalidParameter { ref param, .. } if param == "max_connections"),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn invalid_ssl_and_timeout_are_rejected() {
        let err =
            QuackPoolConfig::from_params(&params(&[("uri", "localhost:9494"), ("ssl", "maybe")]))
                .unwrap_err();
        assert!(matches!(err, Error::InvalidParameter { ref param, .. } if param == "ssl"));

        let err = QuackPoolConfig::from_params(&params(&[
            ("uri", "localhost:9494"),
            ("timeout", "soon"),
        ]))
        .unwrap_err();
        assert!(matches!(err, Error::InvalidParameter { ref param, .. } if param == "timeout"));
    }

    #[test]
    fn ssl_accepts_common_spellings() {
        assert!(parse_bool("ssl", "TRUE").unwrap());
        assert!(!parse_bool("ssl", "off").unwrap());
        assert!(parse_bool("ssl", "").is_err());
    }
}
