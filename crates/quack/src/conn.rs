use std::any::Any;

use arrow::array::{Array, StringArray};
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::error::DataFusionError;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::sql::TableReference;
use futures::{StreamExt, TryStreamExt};
use quack_protocol::{QuackError, QuackPool, Value};
use snafu::prelude::*;

use datafusion_table_providers_common::sql::db_connection_pool::dbconnection::{
    AsyncDbConnection, DbConnection, Error, Result, UnableToGetSchemaSnafu,
    UnableToGetSchemasSnafu, UnableToGetTablesSnafu,
};

/// A connection handle over a shared [`QuackPool`].
///
/// Each query leases one of the pool's server sessions for exactly as long as
/// its result stream is alive, so several `QuackConnection`s (or several
/// queries on one) run concurrently up to the pool's `max_connections`.
///
/// Only the async connection trait is implemented: Quack is an HTTP protocol
/// with an async-only client, and there is no blocking variant to offer.
#[derive(Clone)]
pub struct QuackConnection {
    pool: QuackPool,
}

impl std::fmt::Debug for QuackConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuackConnection").finish_non_exhaustive()
    }
}

impl QuackConnection {
    #[must_use]
    pub fn new(pool: QuackPool) -> Self {
        Self { pool }
    }

    /// Runs `sql` and collects the first column as strings; `NULL`s are skipped.
    async fn string_column(&self, sql: &str) -> Result<Vec<String>> {
        let (_, batches) = self.pool.query(sql, None).await?.into_record_batches()?;
        let batches: Vec<_> = batches.try_collect().await?;

        let mut values = Vec::new();
        for batch in &batches {
            let column = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    format!(
                        "expected a VARCHAR column from '{sql}', got {}",
                        batch.column(0).data_type()
                    )
                })?;
            values.extend(column.iter().flatten().map(str::to_string));
        }
        Ok(values)
    }
}

impl DbConnection<QuackPool, ()> for QuackConnection {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_async(&self) -> Option<&dyn AsyncDbConnection<QuackPool, ()>> {
        Some(self)
    }
}

#[async_trait]
impl AsyncDbConnection<QuackPool, ()> for QuackConnection {
    fn new(conn: QuackPool) -> Self
    where
        Self: Sized,
    {
        Self::new(conn)
    }

    async fn tables(&self, schema: &str) -> Result<Vec<String>, Error> {
        let sql = format!(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = {} ORDER BY table_name",
            quote_literal(schema)
        );
        self.string_column(&sql)
            .await
            .context(UnableToGetTablesSnafu)
    }

    async fn schemas(&self) -> Result<Vec<String>, Error> {
        let sql = "SELECT schema_name FROM information_schema.schemata \
                   WHERE schema_name NOT IN ('information_schema', 'pg_catalog') \
                   ORDER BY schema_name";
        self.string_column(sql)
            .await
            .context(UnableToGetSchemasSnafu)
    }

    /// Infers the Arrow schema by preparing `SELECT * FROM <table> LIMIT 0`.
    ///
    /// Quack returns the column definitions at prepare time, so the schema is
    /// exact (the same mapping every scan uses) even though no rows come back.
    async fn get_schema(&self, table_reference: &TableReference) -> Result<SchemaRef, Error> {
        let sql = format!(
            "SELECT * FROM {} LIMIT 0",
            table_reference.to_quoted_string()
        );

        let result = match self.pool.query(&sql, None).await {
            Ok(result) => result,
            Err(source) if is_undefined_table(&source) => {
                return Err(Error::UndefinedTable {
                    table_name: table_reference.to_string(),
                    source: Box::new(source),
                });
            }
            Err(source) => {
                return Err(source).boxed().context(UnableToGetSchemaSnafu);
            }
        };

        let (schema, mut batches) = result
            .into_record_batches()
            .boxed()
            .context(UnableToGetSchemaSnafu)?;

        // Drain (LIMIT 0 yields nothing) so the leased session goes back to
        // the pool cleanly instead of being abandoned mid-fetch.
        while let Some(batch) = batches.next().await {
            batch.boxed().context(UnableToGetSchemaSnafu)?;
        }

        Ok(schema)
    }

    async fn query_arrow(
        &self,
        sql: &str,
        _params: &[()],
        _projected_schema: Option<SchemaRef>,
    ) -> Result<SendableRecordBatchStream> {
        tracing::debug!(sql, "quack query");
        let (schema, batches) = self.pool.query(sql, None).await?.into_record_batches()?;

        // The stream keeps its pooled session until drained or dropped, which
        // is exactly the lifetime of the DataFusion partition reading it.
        let stream = batches.map_err(|e| DataFusionError::External(Box::new(e)));
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    async fn execute(&self, sql: &str, _params: &[()]) -> Result<u64> {
        // DuckDB answers DML with a single-row, single-column `Count` result;
        // DDL yields no rows. Anything else is drained and reported as 0.
        let values = self.pool.values(sql).await?;
        Ok(match values.as_slice() {
            [Value::Int(n)] => u64::try_from(*n).unwrap_or(0),
            [Value::UInt(n)] => *n,
            _ => 0,
        })
    }
}

/// Renders `value` as a single-quoted DuckDB string literal.
fn quote_literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// Whether a prepare failure means the table does not exist on the server.
///
/// DuckDB reports `Catalog Error: Table with name x does not exist!`; the Quack
/// server forwards it without the `Catalog Error:` prefix, so match the body.
fn is_undefined_table(error: &QuackError) -> bool {
    match error {
        QuackError::Server(message) => {
            message.contains("does not exist")
                && (message.contains("Table with name") || message.contains("Catalog Error"))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_are_escaped() {
        assert_eq!(quote_literal("main"), "'main'");
        assert_eq!(quote_literal("o'clock"), "'o''clock'");
    }

    #[test]
    fn missing_table_is_detected_from_server_message() {
        // As DuckDB phrases it, and as the Quack server forwards it (no prefix).
        let missing = QuackError::Server(
            "Catalog Error: Table with name nope does not exist!\nDid you mean \"pg_type\"?".into(),
        );
        assert!(is_undefined_table(&missing));
        let forwarded = QuackError::Server(
            "Table with name nope does not exist!\nDid you mean \"pg_type\"?\n\nLINE 1: SELECT * FROM nope LIMIT 0".into(),
        );
        assert!(is_undefined_table(&forwarded));

        let other = QuackError::Server("Binder Error: column \"x\" not found".into());
        assert!(!is_undefined_table(&other));

        let transport = QuackError::Protocol("Catalog Error: does not exist".into());
        assert!(!is_undefined_table(&transport));
    }
}
