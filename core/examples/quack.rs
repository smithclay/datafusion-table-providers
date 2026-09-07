use std::{collections::HashMap, sync::Arc};

use datafusion::prelude::SessionContext;
use datafusion::sql::TableReference;
use datafusion_table_providers::{
    quack::QuackTableFactory, sql::db_connection_pool::quackpool::QuackConnectionPool,
    util::secrets::to_secret_map,
};

/// Queries a table served by DuckDB's Quack remote protocol.
///
/// Start a server first (needs the `quack` DuckDB extension):
///
/// ```sh
/// tail -f /dev/null | duckdb -init /dev/null \
///   -cmd "INSTALL quack; LOAD quack; CALL quack_serve('quack:127.0.0.1:9494', token = 'super_secret');"
/// ```
///
/// then `cargo run -p datafusion-table-providers --example quack --features quack`.
#[tokio::main]
async fn main() {
    let params = to_secret_map(HashMap::from([
        ("uri".to_string(), "quack:127.0.0.1:9494".to_string()),
        ("auth_token".to_string(), "super_secret".to_string()),
        // One server session per concurrent DataFusion partition/scan.
        ("max_connections".to_string(), "4".to_string()),
    ]));

    let pool = Arc::new(
        QuackConnectionPool::new(params)
            .await
            .expect("unable to connect to the Quack server"),
    );

    // Use the raw Quack pool for anything the provider does not cover, such as DDL.
    let quack = pool.pool();
    quack
        .execute(
            "CREATE OR REPLACE TABLE companies AS \
             SELECT * FROM (VALUES (1, 'Acme Corporation'), (2, 'Widget Inc.'), (3, 'Gizmo Corp.')) \
             AS t(id, name)",
            None,
        )
        .await
        .expect("create table");

    let ctx = SessionContext::new();
    let table_factory = QuackTableFactory::new(Arc::clone(&pool));

    let companies = table_factory
        .table_provider(TableReference::bare("companies"))
        .await
        .expect("table provider");
    ctx.register_table("companies", companies)
        .expect("register table");

    // Filters, projections and limits are pushed down as DuckDB SQL; with the
    // default `federation` feature the whole query is.
    let df = ctx
        .sql("SELECT id, name FROM companies WHERE id > 1 ORDER BY id")
        .await
        .expect("select failed");
    df.show().await.expect("show failed");

    pool.close().await.expect("close pool");
}
