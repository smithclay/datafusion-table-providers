//! DataFusion table provider for DuckDB's experimental Quack remote protocol.
//!
//! A [`QuackConnectionPool`] wraps a [`quack_protocol::QuackPool`] (several
//! server sessions, one per concurrent scan) and implements the common
//! [`DbConnectionPool`] / [`AsyncDbConnection`] traits. Tables are plain
//! [`SqlTable`]s configured with the DuckDB unparser dialect, so filter,
//! projection and limit pushdown - and, with the `federation` feature, whole
//! query pushdown - work exactly like the in-process DuckDB provider.
//!
//! Quack only offers an async, HTTP-based client, so only the async half of the
//! connection traits is implemented (`as_sync()` returns `None`).
//!
//! [`DbConnectionPool`]: datafusion_table_providers_common::sql::db_connection_pool::DbConnectionPool
//! [`AsyncDbConnection`]: datafusion_table_providers_common::sql::db_connection_pool::dbconnection::AsyncDbConnection

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::{Session, TableProviderFactory};
use datafusion::datasource::TableProvider;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::CreateExternalTable;
use datafusion::sql::unparser::dialect::{Dialect, DuckDBDialect};
use datafusion::sql::TableReference;
use quack_protocol::QuackPool;
use secrecy::SecretString;
use snafu::prelude::*;

use datafusion_table_providers_common::sql::db_connection_pool::DbConnectionPool;
use datafusion_table_providers_common::sql::sql_provider_datafusion::{self, SqlTable};

use crate::pool::QuackConnectionPool;

pub mod conn;
pub mod pool;

/// Re-export of the underlying client crate so callers can match versions and
/// reach the raw [`QuackPool`] for DDL/DML the provider itself does not cover.
pub use quack_protocol;

/// The table type produced by this provider: the common `SqlTable` over a
/// [`QuackPool`]. Quack queries carry no bind parameters, hence `()`.
pub type QuackSqlTable = SqlTable<QuackPool, ()>;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Unable to create Quack connection pool: {source}"))]
    UnableToCreateConnectionPool { source: pool::Error },

    #[snafu(display("Unable to create table provider: {source}"))]
    UnableToCreateTableProvider {
        source: sql_provider_datafusion::Error,
    },

    #[cfg(feature = "federation")]
    #[snafu(display("Unable to create federated table provider: {source}"))]
    UnableToCreateFederatedTableProvider { source: DataFusionError },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The unparser dialect used for every SQL statement sent to a Quack server.
///
/// Quack speaks DuckDB SQL over the wire, so this is the same dialect the
/// in-process DuckDB provider uses.
#[must_use]
pub fn dialect() -> Arc<dyn Dialect + Send + Sync> {
    Arc::new(DuckDBDialect::new())
}

/// Creates read-only table providers for tables reachable through a Quack
/// connection pool.
pub struct QuackTableFactory {
    pool: Arc<QuackConnectionPool>,
}

impl QuackTableFactory {
    #[must_use]
    pub fn new(pool: impl Into<Arc<QuackConnectionPool>>) -> Self {
        Self { pool: pool.into() }
    }

    /// Infers the table's schema from the server and returns a `SqlTable` that
    /// pushes projections, filters and limits down as DuckDB SQL.
    ///
    /// Prefer [`table_provider`](Self::table_provider) unless the un-federated
    /// table is needed (for example to compose it into another provider).
    pub async fn sql_table(
        &self,
        table_reference: impl Into<TableReference>,
    ) -> Result<QuackSqlTable> {
        let dyn_pool =
            Arc::clone(&self.pool) as Arc<dyn DbConnectionPool<QuackPool, ()> + Send + Sync>;
        let table = SqlTable::new("quack", &dyn_pool, table_reference)
            .await
            .context(UnableToCreateTableProviderSnafu)?
            .with_dialect(dialect());
        Ok(table)
    }

    /// Creates a table provider for `table_reference`, wrapped for
    /// `datafusion-federation` when the `federation` feature is enabled.
    pub async fn table_provider(
        &self,
        table_reference: impl Into<TableReference>,
    ) -> Result<Arc<dyn TableProvider + 'static>> {
        let table = Arc::new(self.sql_table(table_reference).await?);

        #[cfg(feature = "federation")]
        let table = Arc::new(
            table
                .create_federated_table_provider()
                .context(UnableToCreateFederatedTableProviderSnafu)?,
        );

        Ok(table)
    }
}

/// `CREATE EXTERNAL TABLE` support.
///
/// ```sql
/// CREATE EXTERNAL TABLE users
///   STORED AS quack
///   LOCATION 'main.users'
///   OPTIONS ('quack.uri' 'quack:127.0.0.1:9494', 'quack.auth_token' 'super_secret');
/// ```
///
/// `LOCATION` names the remote table; when it is empty the DataFusion table
/// name is used. Options are the keys of [`QuackConnectionPool::new`] under
/// the `quack.` namespace (unprefixed keys, which DataFusion files under
/// `format.`, are accepted too). Every statement opens its own pool; use
/// [`QuackTableFactory`] to share one pool across many tables.
#[derive(Debug, Default)]
pub struct QuackTableProviderFactory {}

impl QuackTableProviderFactory {
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

/// Maps `CREATE EXTERNAL TABLE ... OPTIONS (...)` keys onto pool parameters.
///
/// DataFusion lowercases keys and prefixes any un-namespaced key with
/// `format.`; the provider's own namespace is `quack.`.
fn pool_params(options: &HashMap<String, String>) -> HashMap<String, SecretString> {
    options
        .iter()
        .map(|(key, value)| {
            let key = key
                .strip_prefix("quack.")
                .or_else(|| key.strip_prefix("format."))
                .unwrap_or(key);
            (key.to_string(), SecretString::from(value.clone()))
        })
        .collect()
}

/// Resolves the remote table a `CREATE EXTERNAL TABLE` statement refers to.
fn remote_table_reference(cmd: &CreateExternalTable) -> TableReference {
    if cmd.location.trim().is_empty() {
        TableReference::from(cmd.name.to_string())
    } else {
        TableReference::from(cmd.location.as_str())
    }
}

#[async_trait::async_trait]
impl TableProviderFactory for QuackTableProviderFactory {
    async fn create(
        &self,
        _state: &dyn Session,
        cmd: &CreateExternalTable,
    ) -> datafusion::common::Result<Arc<dyn TableProvider>> {
        let pool = QuackConnectionPool::new(pool_params(&cmd.options))
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        QuackTableFactory::new(Arc::new(pool))
            .table_provider(remote_table_reference(cmd))
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::Schema;
    use datafusion::common::DFSchema;

    fn create_cmd(name: &str, location: &str) -> CreateExternalTable {
        CreateExternalTable {
            schema: Arc::new(DFSchema::try_from(Schema::empty()).expect("empty schema")),
            name: TableReference::from(name),
            location: location.to_string(),
            file_type: "quack".to_string(),
            table_partition_cols: vec![],
            if_not_exists: false,
            or_replace: false,
            temporary: false,
            definition: None,
            order_exprs: vec![],
            unbounded: false,
            options: HashMap::new(),
            constraints: Default::default(),
            column_defaults: HashMap::new(),
        }
    }

    #[test]
    fn location_names_the_remote_table() {
        let cmd = create_cmd("local_name", "main.remote_table");
        assert_eq!(
            remote_table_reference(&cmd),
            TableReference::partial("main", "remote_table")
        );
    }

    #[test]
    fn empty_location_falls_back_to_table_name() {
        let cmd = create_cmd("users", "  ");
        assert_eq!(remote_table_reference(&cmd), TableReference::bare("users"));
    }

    #[test]
    fn option_keys_are_unprefixed() {
        use secrecy::ExposeSecret;

        let options = HashMap::from([
            ("quack.uri".to_string(), "localhost:9494".to_string()),
            ("format.auth_token".to_string(), "token".to_string()),
            ("max_connections".to_string(), "2".to_string()),
        ]);
        let params = pool_params(&options);
        let get = |key: &str| params.get(key).map(|v| v.expose_secret().to_string());
        assert_eq!(get("uri").as_deref(), Some("localhost:9494"));
        assert_eq!(get("auth_token").as_deref(), Some("token"));
        assert_eq!(get("max_connections").as_deref(), Some("2"));
        assert_eq!(params.len(), 3);
    }

    #[test]
    fn dialect_quotes_identifiers_like_duckdb() {
        assert_eq!(dialect().identifier_quote_style("x"), Some('"'));
    }
}
