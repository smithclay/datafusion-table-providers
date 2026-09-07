//! Live tests against a Quack server. They skip unless `QUACK_SERVER_URI` is
//! set (optionally with `QUACK_AUTH_TOKEN`). Start a server with:
//!
//! ```sh
//! tail -f /dev/null | duckdb -init /dev/null \
//!   -cmd "INSTALL quack; LOAD quack; CALL quack_serve('quack:127.0.0.1:9495', token = 'super_secret');" &
//! QUACK_SERVER_URI=quack:127.0.0.1:9495 QUACK_AUTH_TOKEN=super_secret \
//!   cargo test -p datafusion-table-providers-quack
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::DataType;
use arrow::util::pretty::pretty_format_batches;
use datafusion::prelude::SessionContext;
use datafusion::sql::TableReference;
use datafusion_table_providers_common::sql::db_connection_pool::dbconnection::Error as DbError;
use datafusion_table_providers_common::sql::db_connection_pool::DbConnectionPool;
use datafusion_table_providers_quack::pool::QuackConnectionPool;
use datafusion_table_providers_quack::{QuackTableFactory, QuackTableProviderFactory};
use secrecy::SecretString;

/// The tests share fixture tables on one server, and DuckDB rejects concurrent
/// catalog writes with a write-write conflict, so they run one at a time.
static FIXTURES: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn connect() -> Option<(
    Arc<QuackConnectionPool>,
    tokio::sync::MutexGuard<'static, ()>,
)> {
    let Ok(uri) = std::env::var("QUACK_SERVER_URI") else {
        eprintln!("QUACK_SERVER_URI not set; skipping Quack integration test");
        return None;
    };
    let guard = FIXTURES.lock().await;
    let mut params = HashMap::from([("uri".to_string(), SecretString::from(uri))]);
    if let Ok(token) = std::env::var("QUACK_AUTH_TOKEN") {
        params.insert("auth_token".to_string(), SecretString::from(token));
    }
    params.insert("max_connections".to_string(), SecretString::from("4"));
    let pool = QuackConnectionPool::new(params)
        .await
        .expect("connect to Quack server");
    Some((Arc::new(pool), guard))
}

async fn create_fixtures(pool: &QuackConnectionPool) {
    let quack = pool.pool();
    for sql in [
        "CREATE OR REPLACE TABLE quack_it_users (id INTEGER, name VARCHAR, score DOUBLE, joined DATE)",
        "INSERT INTO quack_it_users VALUES \
            (1, 'alice', 1.5, DATE '2024-01-01'), \
            (2, 'bob', 2.25, DATE '2024-02-02'), \
            (3, 'carol', NULL, DATE '2024-03-03')",
        "CREATE OR REPLACE TABLE quack_it_orders (order_id INTEGER, user_id INTEGER, amount DECIMAL(10,2))",
        "INSERT INTO quack_it_orders VALUES (10, 1, 9.99), (11, 2, 20.00), (12, 2, 5.01)",
    ] {
        quack.execute(sql, None).await.unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
}

async fn drop_fixtures(pool: &QuackConnectionPool) {
    let quack = pool.pool();
    for table in ["quack_it_users", "quack_it_orders"] {
        let _ = quack
            .execute(&format!("DROP TABLE IF EXISTS {table}"), None)
            .await;
    }
}

/// A session with `datafusion-federation`'s optimizer installed, so joins and
/// aggregates over tables from one pool are rewritten into a single remote
/// query. Without the feature this is a plain session and only per-scan
/// pushdown (projection/filter/limit) applies.
fn session() -> SessionContext {
    #[cfg(feature = "federation")]
    {
        SessionContext::new_with_state(datafusion_federation::default_session_state())
    }
    #[cfg(not(feature = "federation"))]
    {
        SessionContext::new()
    }
}

fn rows(batches: &[arrow::array::RecordBatch]) -> String {
    pretty_format_batches(batches).expect("format").to_string()
}

#[tokio::test]
async fn query_quack_tables_through_datafusion() {
    let Some((pool, _guard)) = connect().await else {
        return;
    };
    create_fixtures(&pool).await;

    let factory = QuackTableFactory::new(Arc::clone(&pool));
    let users = factory
        .table_provider(TableReference::bare("quack_it_users"))
        .await
        .expect("users provider");
    let orders = factory
        .table_provider(TableReference::partial("main", "quack_it_orders"))
        .await
        .expect("orders provider");

    // Schema inference comes from the server's prepare-time column definitions.
    let schema = users.schema();
    assert_eq!(schema.field(0).name(), "id");
    assert_eq!(schema.field(0).data_type(), &DataType::Int32);
    assert_eq!(schema.field(1).data_type(), &DataType::Utf8);
    assert_eq!(schema.field(2).data_type(), &DataType::Float64);
    assert_eq!(schema.field(3).data_type(), &DataType::Date32);

    let ctx = session();
    ctx.register_table("users", users).unwrap();
    ctx.register_table("orders", orders).unwrap();

    // Projection + filter + limit pushdown on one table.
    let batches = ctx
        .sql("SELECT id, name FROM users WHERE id >= 2 ORDER BY id LIMIT 1")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        rows(&batches),
        "+----+------+\n\
         | id | name |\n\
         +----+------+\n\
         | 2  | bob  |\n\
         +----+------+"
    );

    // NULL handling and a DATE column.
    let batches = ctx
        .sql("SELECT name, joined FROM users WHERE score IS NULL")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        rows(&batches),
        "+-------+------------+\n\
         | name  | joined     |\n\
         +-------+------------+\n\
         | carol | 2024-03-03 |\n\
         +-------+------------+"
    );

    // A join between two tables on the same pool; with federation enabled the
    // whole thing runs as one DuckDB query on the server.
    let join_sql = "SELECT u.name, SUM(o.amount) AS total \
                    FROM users u JOIN orders o ON u.id = o.user_id \
                    GROUP BY u.name ORDER BY u.name";
    let batches = ctx.sql(join_sql).await.unwrap().collect().await.unwrap();
    assert_eq!(
        rows(&batches),
        "+-------+-------+\n\
         | name  | total |\n\
         +-------+-------+\n\
         | alice | 9.99  |\n\
         | bob   | 25.01 |\n\
         +-------+-------+"
    );

    #[cfg(feature = "federation")]
    {
        let plan = ctx
            .sql(&format!("EXPLAIN {join_sql}"))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let plan = rows(&plan);
        assert!(
            plan.contains("VirtualExecutionPlan"),
            "expected the join to be federated into a single remote query:\n{plan}"
        );
    }

    drop_fixtures(&pool).await;
    pool.close().await.expect("close pool");
}

#[tokio::test]
async fn connection_catalog_helpers() {
    let Some((pool, _guard)) = connect().await else {
        return;
    };
    create_fixtures(&pool).await;

    let conn = pool.connect().await.expect("connection");
    let conn = conn.as_async().expect("quack connections are async");

    let schemas = conn.schemas().await.expect("schemas");
    assert!(schemas.iter().any(|s| s == "main"), "{schemas:?}");

    let tables = conn.tables("main").await.expect("tables");
    assert!(tables.contains(&"quack_it_users".to_string()), "{tables:?}");
    assert!(
        tables.contains(&"quack_it_orders".to_string()),
        "{tables:?}"
    );

    let err = conn
        .get_schema(&TableReference::bare("quack_it_definitely_missing"))
        .await
        .expect_err("missing table must fail");
    assert!(matches!(err, DbError::UndefinedTable { .. }), "{err}");

    let affected = conn
        .execute(
            "UPDATE quack_it_users SET score = 0 WHERE score IS NULL",
            &[],
        )
        .await
        .expect("update");
    assert_eq!(affected, 1);

    drop_fixtures(&pool).await;
}

#[tokio::test]
async fn create_external_table() {
    let Some((pool, _guard)) = connect().await else {
        return;
    };
    create_fixtures(&pool).await;

    let ctx = session();
    ctx.state_ref().write().table_factories_mut().insert(
        "QUACK".to_string(),
        Arc::new(QuackTableProviderFactory::new()),
    );

    let mut options = format!(
        "'quack.uri' '{}'",
        std::env::var("QUACK_SERVER_URI").expect("checked by connect()")
    );
    if let Ok(token) = std::env::var("QUACK_AUTH_TOKEN") {
        options.push_str(&format!(", 'quack.auth_token' '{token}'"));
    }
    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE ext_users STORED AS quack LOCATION 'quack_it_users' OPTIONS ({options})"
    ))
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let batches = ctx
        .sql("SELECT count(*) AS n FROM ext_users")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        rows(&batches),
        "+---+\n\
         | n |\n\
         +---+\n\
         | 3 |\n\
         +---+"
    );

    drop_fixtures(&pool).await;
}
