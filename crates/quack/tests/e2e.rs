//! End-to-end validation harness for the Quack provider.
//!
//! Everything here skips unless `QUACK_SERVER_URI` is set (with
//! `QUACK_AUTH_TOKEN` when the server has a token). The tests that build a
//! 5M-row fact table additionally require `QUACK_E2E=1`. Timings are printed,
//! never asserted. Run with:
//!
//! ```sh
//! QUACK_SERVER_URI=quack:127.0.0.1:9495 QUACK_AUTH_TOKEN=super_secret QUACK_E2E=1 \
//!   cargo test -p datafusion-table-providers-quack --test e2e -- --nocapture --test-threads=1
//! ```
//!
//! Local `duckdb` CLI baselines run when a `duckdb` binary is on `PATH`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use datafusion::sql::TableReference;
use datafusion_table_providers_quack::pool::QuackConnectionPool;
use datafusion_table_providers_quack::{QuackTableFactory, QuackTableProviderFactory};
use futures::{StreamExt, TryStreamExt};
use quack_protocol::{
    sql_literal, QuackClient, QuackClientOptions, QuackPool, SqlParameter, Value,
};
use secrecy::SecretString;
use tokio::task::JoinSet;
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

// ---------------------------------------------------------------------------
// Environment / setup
// ---------------------------------------------------------------------------

static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn server_uri() -> Option<String> {
    let uri = std::env::var("QUACK_SERVER_URI").ok();
    if uri.is_none() {
        eprintln!("QUACK_SERVER_URI not set; skipping");
    }
    uri
}

fn e2e_enabled() -> bool {
    let enabled = std::env::var("QUACK_E2E")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !enabled {
        eprintln!("QUACK_E2E=1 not set; skipping slow e2e test");
    }
    enabled
}

fn token() -> Option<String> {
    std::env::var("QUACK_AUTH_TOKEN").ok()
}

fn params(
    uri: &str,
    max_connections: usize,
    extra: &[(&str, &str)],
) -> HashMap<String, SecretString> {
    let mut p = HashMap::from([
        ("uri".to_string(), SecretString::from(uri.to_string())),
        (
            "max_connections".to_string(),
            SecretString::from(max_connections.to_string()),
        ),
    ]);
    if let Some(t) = token() {
        p.insert("auth_token".to_string(), SecretString::from(t));
    }
    for (k, v) in extra {
        p.insert((*k).to_string(), SecretString::from((*v).to_string()));
    }
    p
}

async fn connect(uri: &str, max_connections: usize) -> Arc<QuackConnectionPool> {
    Arc::new(
        QuackConnectionPool::new(params(uri, max_connections, &[]))
            .await
            .expect("connect"),
    )
}

async fn control_client(uri: &str) -> QuackClient {
    QuackClient::connect(
        uri,
        QuackClientOptions {
            auth_token: token(),
            ..Default::default()
        },
    )
    .await
    .expect("control client")
}

fn session() -> SessionContext {
    SessionContext::new_with_state(datafusion_federation::default_session_state())
}

fn quoted(v: &str) -> String {
    sql_literal(&SqlParameter::from(v)).expect("literal")
}

fn print_env(pool: &QuackConnectionPool) {
    let quack = pool.pool();
    let info = quack.info();
    println!(
        "ENV duckdb_cli={} server_duckdb={:?} server_platform={:?} quack_protocol_version={:?} heartbeat={:?} client_arch={} threads={}",
        duckdb_cli_version().unwrap_or_else(|| "n/a".into()),
        info.and_then(|i| i.server_duckdb_version.clone()),
        info.and_then(|i| i.server_platform.clone()),
        info.and_then(|i| i.quack_version),
        info.and_then(|i| i.heartbeat_timeout),
        std::env::consts::ARCH,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
    );
}

// ---------------------------------------------------------------------------
// Round-trip counting via quack_protocol's tracing events
// ---------------------------------------------------------------------------

static PREPARES: AtomicUsize = AtomicUsize::new(0);
static FETCHES: AtomicUsize = AtomicUsize::new(0);
static DISCONNECTS: AtomicUsize = AtomicUsize::new(0);
static PREPARE_SQL: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

struct EventCounter;

impl<S: Subscriber> Layer<S> for EventCounter {
    fn enabled(&self, meta: &tracing::Metadata<'_>, _ctx: Context<'_, S>) -> bool {
        meta.target().starts_with("quack_protocol")
    }

    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        struct MessageVisitor(String, String);
        impl tracing::field::Visit for MessageVisitor {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                match field.name() {
                    "message" => self.0 = format!("{value:?}"),
                    "sql" => self.1 = format!("{value:?}"),
                    _ => {}
                }
            }
        }
        let mut v = MessageVisitor(String::new(), String::new());
        event.record(&mut v);
        // The harness's own control-plane queries are not provider traffic.
        // quack_protocol logs the SQL elided (`"SELECT act...rt = 19702"`),
        // so match on the prefix of `SELECT active_connections ...` and of
        // `CALL quack_serve/quack_stop`.
        let sql = v.1.trim_start_matches('"');
        if sql.starts_with("SELECT act") || sql.starts_with("CALL quack") {
            return;
        }
        if v.0.contains("quack FETCH completed") {
            FETCHES.fetch_add(1, Ordering::Relaxed);
        } else if v.0.contains("quack PREPARE completed") {
            PREPARES.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut log) = PREPARE_SQL.lock() {
                log.push(v.1.chars().take(120).collect());
                if log.len() > 8 {
                    log.remove(0);
                }
            }
        } else if v.0.contains("session closed on drop") {
            DISCONNECTS.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn init_tracing() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let subscriber = tracing_subscriber::registry().with(EventCounter);
        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

#[derive(Debug, Clone, Copy)]
struct Counts {
    prepares: usize,
    fetches: usize,
}

fn counts() -> Counts {
    Counts {
        prepares: PREPARES.load(Ordering::Relaxed),
        fetches: FETCHES.load(Ordering::Relaxed),
    }
}

fn delta(before: Counts) -> Counts {
    let now = counts();
    Counts {
        prepares: now.prepares - before.prepares,
        fetches: now.fetches - before.fetches,
    }
}

fn cpu_seconds() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    tv(usage.ru_utime) + tv(usage.ru_stime)
}

fn peak_rss_mb() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    // macOS reports bytes, Linux kilobytes.
    if cfg!(target_os = "macos") {
        usage.ru_maxrss as f64 / (1024.0 * 1024.0)
    } else {
        usage.ru_maxrss as f64 / 1024.0
    }
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

fn fmt(batches: &[RecordBatch]) -> String {
    pretty_format_batches(batches).expect("format").to_string()
}

async fn df_collect(ctx: &SessionContext, sql: &str) -> Result<Vec<RecordBatch>, String> {
    let df = ctx.sql(sql).await.map_err(|e| e.to_string())?;
    df.collect().await.map_err(|e| e.to_string())
}

async fn df_rows(ctx: &SessionContext, sql: &str) -> String {
    match df_collect(ctx, sql).await {
        Ok(b) => fmt(&b),
        Err(e) => format!("ERROR: {e}"),
    }
}

async fn explain(ctx: &SessionContext, sql: &str) -> String {
    let batches = df_collect(ctx, &format!("EXPLAIN {sql}"))
        .await
        .expect("explain");
    // Only the physical plan column, trimmed.
    let mut out = String::new();
    for b in &batches {
        let kinds = b.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let plans = b.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..b.num_rows() {
            if kinds.value(i) == "physical_plan" {
                out.push_str(plans.value(i));
            }
        }
    }
    out.trim_end().to_string()
}

async fn raw_batches(pool: &QuackPool, sql: &str) -> Result<Vec<RecordBatch>, String> {
    let (_, stream) = pool
        .query(sql, None)
        .await
        .map_err(|e| e.to_string())?
        .into_record_batches()
        .map_err(|e| e.to_string())?;
    stream.try_collect().await.map_err(|e| e.to_string())
}

async fn raw_rows(pool: &QuackPool, sql: &str) -> String {
    match raw_batches(pool, sql).await {
        Ok(b) => fmt(&b),
        Err(e) => format!("ERROR: {e}"),
    }
}

fn batch_bytes(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.get_array_memory_size()).sum()
}

fn batch_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

// ---------------------------------------------------------------------------
// duckdb CLI baseline
// ---------------------------------------------------------------------------

fn duckdb_cli_version() -> Option<String> {
    let out = Command::new("duckdb").arg("--version").output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn scratch_dir() -> PathBuf {
    std::env::var("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
}

fn scratch_db() -> PathBuf {
    scratch_dir().join("quack_e2e_fact.duckdb")
}

fn types_db() -> PathBuf {
    scratch_dir().join("quack_e2e_types.duckdb")
}

/// Runs `sql` in the duckdb CLI against the fact scratch file DB; returns
/// (csv output, wall time including process start-up).
fn duckdb_cli(sql: &str) -> Option<(String, Duration)> {
    duckdb_cli_on(&scratch_db(), sql)
}

fn duckdb_cli_on(db: &PathBuf, sql: &str) -> Option<(String, Duration)> {
    let started = Instant::now();
    let out = Command::new("duckdb")
        .arg(db)
        .arg("-csv")
        .arg("-c")
        .arg(sql)
        .output()
        .ok()?;
    let elapsed = started.elapsed();
    if !out.status.success() {
        return Some((
            format!("CLI ERROR: {}", String::from_utf8_lossy(&out.stderr)),
            elapsed,
        ));
    }
    Some((String::from_utf8_lossy(&out.stdout).to_string(), elapsed))
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Deterministic variant of the requested fact table (`random()` replaced by
/// a hash so DataFusion, the raw pool and the CLI can be compared row for row).
const FACT_DDL: &str = "CREATE TABLE fact AS SELECT range AS id, range % 1000 AS dim_id, \
    ((hash(range) % 100000) / 100.0)::DOUBLE AS amount, ('s' || (range % 97)) AS label, \
    (DATE '2020-01-01' + INTERVAL (range % 730) DAY)::DATE AS d FROM range(5000000)";
const DIM_DDL: &str = "CREATE TABLE dim AS SELECT range AS dim_id, 'dim_' || range AS dim_name, \
    range % 7 AS region FROM range(1000)";

async fn ensure_fact(pool: &QuackPool) {
    // Reuse the fixture only if it is the current shape (`d` must be a DATE).
    let exists = pool
        .values(
            "SELECT count(*) FROM information_schema.columns \
             WHERE (table_name = 'fact' AND column_name = 'd' AND data_type = 'DATE') \
                OR (table_name = 'dim' AND column_name = 'region')",
        )
        .await
        .expect("check");
    if matches!(exists.first(), Some(Value::Int(2))) {
        if duckdb_cli_version().is_some() && !scratch_db().exists() {
            let (out, _) = duckdb_cli(&format!("{FACT_DDL}; {DIM_DDL};")).expect("cli");
            println!("FIXTURE local file DB rebuilt {}", out.trim());
        }
        return;
    }
    let started = Instant::now();
    for sql in [
        "DROP TABLE IF EXISTS fact",
        "DROP TABLE IF EXISTS dim",
        FACT_DDL,
        DIM_DDL,
    ] {
        pool.execute(sql, None)
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    println!(
        "FIXTURE fact(5M)+dim(1000) created on server in {:?}",
        started.elapsed()
    );

    if duckdb_cli_version().is_some() {
        let _ = std::fs::remove_file(scratch_db());
        let started = Instant::now();
        let (out, _) = duckdb_cli(&format!("{FACT_DDL}; {DIM_DDL};")).expect("cli");
        println!(
            "FIXTURE local file DB created in {:?} {}",
            started.elapsed(),
            out.trim()
        );
    }
}

// ===========================================================================
// SEAM 1: type coverage
// ===========================================================================

const TYPES_DDL: &str = "CREATE OR REPLACE TABLE e2e_types (id INTEGER, \
    c_bool BOOLEAN, c_i8 TINYINT, c_i16 SMALLINT, c_i32 INTEGER, c_i64 BIGINT, \
    c_u8 UTINYINT, c_u16 USMALLINT, c_u32 UINTEGER, c_u64 UBIGINT, c_hugeint HUGEINT, \
    c_f32 FLOAT, c_f64 DOUBLE, c_dec18 DECIMAL(18,3), c_dec38 DECIMAL(38,10), \
    c_varchar VARCHAR, c_blob BLOB, c_date DATE, c_time TIME, c_ts TIMESTAMP, \
    c_ts_s TIMESTAMP_S, c_ts_ms TIMESTAMP_MS, c_ts_ns TIMESTAMP_NS, c_tstz TIMESTAMPTZ, \
    c_interval INTERVAL, c_uuid UUID, c_list INTEGER[], c_struct STRUCT(a INTEGER, b VARCHAR), \
    c_list_struct STRUCT(x INTEGER, y DOUBLE)[], c_array INTEGER[3])";

const TYPES_ROWS: &str = "INSERT INTO e2e_types VALUES \
    (1, true, -128, -32768, -2147483648, -9223372036854775808, \
        255, 65535, 4294967295, 18446744073709551615, 170141183460469231731687303715884105727, \
        3.4028235e38, -2.5e-300, 123456789012345.678, 1234567890123456789012345678.1234567890, \
        'ünïcödé ''quoted''', '\\xDE\\xAD\\xBE\\xEF'::BLOB, DATE '1969-12-31', TIME '23:59:59.999999', \
        TIMESTAMP '1969-12-31 23:59:59.999999', TIMESTAMP_S '2000-01-01 00:00:00', \
        TIMESTAMP_MS '2000-01-01 00:00:00.123', TIMESTAMP_NS '2000-01-01 00:00:00.123456789', \
        TIMESTAMPTZ '2000-01-01 12:00:00+02', INTERVAL '1 year 2 months 3 days 04:05:06.789', \
        '550e8400-e29b-41d4-a716-446655440000'::UUID, [1, NULL, 3], {'a': 1, 'b': 'x'}, \
        [{'x': 1, 'y': 1.5}, {'x': NULL, 'y': NULL}], [1, 2, 3]), \
    (2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, \
        NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL), \
    (3, false, 0, 0, 0, 0, 0, 0, 0, 0, -170141183460469231731687303715884105728, \
        'nan'::FLOAT, 'inf'::DOUBLE, -0.001, -0.0000000001, '', ''::BLOB, DATE '2024-02-29', \
        TIME '00:00:00', TIMESTAMP '2024-02-29 12:34:56', TIMESTAMP_S '1970-01-01 00:00:00', \
        TIMESTAMP_MS '1970-01-01 00:00:00', TIMESTAMP_NS '1970-01-01 00:00:00', \
        TIMESTAMPTZ '1970-01-01 00:00:00+00', INTERVAL '0 days', \
        '00000000-0000-0000-0000-000000000000'::UUID, [], {'a': NULL, 'b': NULL}, [], [NULL, NULL, NULL])";

#[tokio::test]
async fn seam_1_type_coverage() {
    let Some(uri) = server_uri() else { return };
    let _guard = LOCK.lock().await;
    init_tracing();
    let pool = connect(&uri, 4).await;
    print_env(&pool);
    let quack = pool.pool();
    for sql in [TYPES_DDL, TYPES_ROWS] {
        quack
            .execute(sql, None)
            .await
            .unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
    let has_cli = duckdb_cli_version().is_some();
    if has_cli {
        let _ = std::fs::remove_file(types_db());
        let (out, _) = duckdb_cli_on(&types_db(), &format!("{TYPES_DDL}; {TYPES_ROWS};")).unwrap();
        if out.contains("ERROR") {
            println!("CLI fixture: {out}");
        }
    }

    let factory = QuackTableFactory::new(Arc::clone(&pool));
    let provider = match factory
        .table_provider(TableReference::bare("e2e_types"))
        .await
    {
        Ok(p) => p,
        Err(e) => {
            println!("TYPES schema-time failure for the whole table: {e}");
            return;
        }
    };
    let schema = provider.schema();
    let ctx = session();
    ctx.register_table("t", provider).unwrap();

    println!("\n===== SEAM 1: type coverage (DataFusion vs duckdb CLI for the same rows) =====");
    for field in schema.fields().iter().skip(1) {
        let col = field.name();
        let sql = format!("SELECT {col} FROM e2e_types ORDER BY id");
        let df = df_rows(&ctx, &format!("SELECT {col} FROM t ORDER BY id")).await;
        let cli = if has_cli {
            duckdb_cli_on(&types_db(), &sql)
                .map(|(o, _)| o)
                .unwrap_or_default()
        } else {
            "n/a".into()
        };
        println!(
            "\n--- {col}: arrow={:?}\nDataFusion:\n{df}\nduckdb -csv:\n{}",
            field.data_type(),
            cli.trim_end()
        );
    }

    // Whole-row sanity: the same SELECT * through DataFusion and through the
    // raw pool must agree (same bridge, but DataFusion adds its own copy/cast).
    let df_all = df_rows(&ctx, "SELECT * FROM t ORDER BY id").await;
    let raw_all = raw_rows(&quack, "SELECT * FROM e2e_types ORDER BY id").await;
    println!(
        "\nSELECT * DataFusion == raw pool: {}",
        if df_all == raw_all { "yes" } else { "NO" }
    );
    if df_all != raw_all {
        println!("DataFusion:\n{df_all}\nraw:\n{raw_all}");
    }

    // DataFusion-side work on the fetched arrays (not pushed down): a join
    // against a MemTable forces the batches through DataFusion operators.
    let mem_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let mem = RecordBatch::try_new(
        Arc::clone(&mem_schema),
        vec![Arc::new(arrow::array::Int32Array::from(vec![1, 3])) as ArrayRef],
    )
    .unwrap();
    ctx.register_table(
        "m",
        Arc::new(MemTable::try_new(mem_schema, vec![vec![mem]]).unwrap()),
    )
    .unwrap();
    println!(
        "\nMemTable join over every column (DataFusion HashJoin on fetched arrays):\n{}",
        df_rows(
            &ctx,
            "SELECT t.* FROM t JOIN m ON t.id = m.id ORDER BY t.id"
        )
        .await
    );

    // Types expected to be awkward, one table each so the failure is isolated.
    println!("\n===== SEAM 1b: awkward types =====");
    let awkward: &[(&str, &str, &str)] = &[
        ("MAP", "MAP(VARCHAR, INTEGER)", "MAP {'k1': 1, 'k2': NULL}"),
        ("ENUM", "ENUM('small', 'large')", "'large'"),
        (
            "UNION",
            "UNION(num INTEGER, str VARCHAR)",
            "union_value(str := 'u')",
        ),
        ("TIMETZ", "TIMETZ", "TIMETZ '12:00:00+05:30'"),
        ("BIT", "BIT", "'1011'::BIT"),
        (
            "VARINT",
            "VARINT",
            "'123456789012345678901234567890'::VARINT",
        ),
        ("JSON", "JSON", "'{\"a\": [1, 2]}'::JSON"),
        ("TIME 24:00", "TIME", "TIME '24:00:00'"),
    ];
    for (name, ty, literal) in awkward {
        let table = format!("e2e_awk_{}", name.to_lowercase().replace([' ', ':'], "_"));
        let ddl = format!("CREATE OR REPLACE TABLE {table} (id INTEGER, v {ty})");
        if let Err(e) = quack.execute(&ddl, None).await {
            println!("{name}: could not create table ({ty}): {e}");
            continue;
        }
        let ins = format!("INSERT INTO {table} VALUES (1, {literal}), (2, NULL)");
        if let Err(e) = quack.execute(&ins, None).await {
            println!("{name}: could not insert {literal}: {e}");
            continue;
        }
        match factory
            .table_provider(TableReference::bare(table.as_str()))
            .await
        {
            Err(e) => println!("{name} ({ty}): SCHEMA-TIME failure: {e}"),
            Ok(p) => {
                let arrow_ty = p.schema().field(1).data_type().clone();
                let ctx = session();
                ctx.register_table("a", p).unwrap();
                let out = df_rows(&ctx, "SELECT v FROM a ORDER BY id").await;
                let kind = if out.starts_with("ERROR") {
                    "QUERY-TIME failure"
                } else {
                    "ok"
                };
                println!("{name} ({ty}) -> arrow {arrow_ty:?}: {kind}\n{out}");
            }
        }
    }
    pool.close().await.ok();
}

// ===========================================================================
// SEAM 2: pushdown correctness (E2E)
// ===========================================================================

#[tokio::test]
async fn seam_2_pushdown_correctness() {
    let Some(uri) = server_uri() else { return };
    if !e2e_enabled() {
        return;
    }
    let _guard = LOCK.lock().await;
    init_tracing();
    let pool = connect(&uri, 4).await;
    let quack = pool.pool();
    ensure_fact(&quack).await;

    let factory = QuackTableFactory::new(Arc::clone(&pool));
    let ctx = session();
    ctx.register_table("fact", factory.table_provider("fact").await.unwrap())
        .unwrap();
    ctx.register_table("dim", factory.table_provider("dim").await.unwrap())
        .unwrap();
    // In-memory DataFusion table for the mixed join.
    let mem_schema = Arc::new(Schema::new(vec![
        Field::new("dim_id", DataType::Int64, false),
        Field::new("bucket", DataType::Utf8, false),
    ]));
    let mem = RecordBatch::try_new(
        Arc::clone(&mem_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec!["one", "two", "three"])) as ArrayRef,
        ],
    )
    .unwrap();
    ctx.register_table(
        "mem",
        Arc::new(MemTable::try_new(mem_schema, vec![vec![mem]]).unwrap()),
    )
    .unwrap();

    println!("\n===== SEAM 2: pushdown correctness =====");
    let cases: &[(&str, &str, Option<&str>)] = &[
        ("count(*)", "SELECT count(*) FROM fact", Some("SELECT count(*) FROM fact")),
        ("projection", "SELECT id, label FROM fact WHERE id < 5 ORDER BY id", Some("SELECT id, label FROM fact WHERE id < 5 ORDER BY id")),
        ("filter on date", "SELECT count(*), min(d), max(d) FROM fact WHERE d BETWEEN DATE '2020-03-01' AND DATE '2020-03-31'", Some("SELECT count(*), min(d), max(d) FROM fact WHERE d BETWEEN DATE '2020-03-01' AND DATE '2020-03-31'")),
        ("filter on date + string", "SELECT count(*) FROM fact WHERE d = DATE '2021-12-30' AND label = 's7'", Some("SELECT count(*) FROM fact WHERE d = DATE '2021-12-30' AND label = 's7'")),
        ("LIMIT", "SELECT id FROM fact LIMIT 3", None),
        ("ORDER BY + LIMIT", "SELECT id, amount FROM fact ORDER BY amount DESC, id LIMIT 3", Some("SELECT id, amount FROM fact ORDER BY amount DESC, id LIMIT 3")),
        ("GROUP BY aggregates", "SELECT dim_id, count(*) AS n, sum(amount) AS s, avg(amount) AS a FROM fact WHERE dim_id < 3 GROUP BY dim_id ORDER BY dim_id", Some("SELECT dim_id, count(*) AS n, sum(amount) AS s, avg(amount) AS a FROM fact WHERE dim_id < 3 GROUP BY dim_id ORDER BY dim_id")),
        ("SUM of INTEGER column", "SELECT sum(id) AS s, sum(dim_id) AS sd FROM fact", Some("SELECT sum(id) AS s, sum(dim_id) AS sd FROM fact")),
        ("quack-quack join", "SELECT d.region, count(*) AS n, sum(f.amount) AS s FROM fact f JOIN dim d ON f.dim_id = d.dim_id WHERE d.region = 3 GROUP BY d.region", Some("SELECT d.region, count(*) AS n, sum(f.amount) AS s FROM fact f JOIN dim d ON f.dim_id = d.dim_id WHERE d.region = 3 GROUP BY d.region")),
        ("quack-MemTable join", "SELECT m.bucket, count(*) AS n FROM fact f JOIN mem m ON f.dim_id = m.dim_id GROUP BY m.bucket ORDER BY m.bucket", Some("SELECT CASE dim_id WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'three' END AS bucket, count(*) AS n FROM fact WHERE dim_id IN (1,2,3) GROUP BY 1 ORDER BY 1")),
    ];
    for (name, sql, local) in cases {
        let before = counts();
        let started = Instant::now();
        let out = df_rows(&ctx, sql).await;
        let elapsed = started.elapsed();
        let d = delta(before);
        let plan = explain(&ctx, sql).await;
        let schema_desc = match ctx.sql(sql).await {
            Ok(df) => df
                .schema()
                .fields()
                .iter()
                .map(|f| format!("{}:{:?}", f.name(), f.data_type()))
                .collect::<Vec<_>>()
                .join(", "),
            Err(e) => e.to_string(),
        };
        println!(
            "\n--- {name}\nSQL: {sql}\nDataFusion result schema: {schema_desc}\nwall={elapsed:?} prepares={} fetches={}\nphysical plan:\n{plan}\nresult:\n{out}",
            d.prepares, d.fetches
        );
        if let Some(local) = local {
            let raw = raw_rows(&quack, local).await;
            // Compare data rows only, ignoring padding: pretty-printing pads
            // cells to the header width, which differs when DuckDB names a
            // column `count_star()` and DataFusion `count(*)`.
            let body = |s: &str| {
                s.lines()
                    .skip(3)
                    .filter(|l| !l.starts_with('+'))
                    .map(|l| l.chars().filter(|c| !c.is_whitespace()).collect::<String>())
                    .collect::<Vec<_>>()
            };
            let same = body(&raw) == body(&out);
            println!(
                "raw DuckDB (same SQL via pool): {}",
                if same {
                    "IDENTICAL rows (header naming may differ)"
                } else {
                    "DIFFERS (float aggregates may differ in the last digits by summation order)"
                }
            );
            if !same {
                println!("{raw}");
            }
            if let Some((cli, _)) = duckdb_cli(local) {
                println!("duckdb -csv:\n{}", cli.trim_end());
            }
        }
    }
    pool.close().await.ok();
}

// ===========================================================================
// SEAM 3: concurrency & pool behaviour (E2E, nested server for counting)
// ===========================================================================

const NESTED_CONCURRENCY_PORT: u16 = 19_701;
const NESTED_CANCEL_PORT: u16 = 19_702;
const NESTED_RESTART_PORT: u16 = 19_703;
const NESTED_UNREACHABLE_PORT: u16 = 19_704;
const NESTED_TIMEOUT_PORT: u16 = 19_705;

fn nested_uri(port: u16) -> String {
    format!("quack:127.0.0.1:{port}")
}

async fn start_nested(control: &QuackClient, port: u16) -> String {
    let uri = nested_uri(port);
    let _ = stop_nested(control, port).await;
    let tok = token()
        .map(|t| format!(", token = {}", quoted(&t)))
        .unwrap_or_default();
    control
        .execute(&format!("CALL quack_serve({}{tok})", quoted(&uri)), None)
        .await
        .expect("quack_serve");
    uri
}

async fn stop_nested(control: &QuackClient, port: u16) -> Result<(), String> {
    control
        .execute(
            &format!("CALL quack_stop({})", quoted(&nested_uri(port))),
            None,
        )
        .await
        .map(|_| ())
        .map_err(|e| e.to_string())
}

async fn active_connections(control: &QuackClient, port: u16) -> u64 {
    let values = control
        .values(&format!(
            "SELECT active_connections FROM quack_server_list() WHERE port = {port}"
        ))
        .await
        .expect("quack_server_list");
    match values.first() {
        Some(Value::UInt(n)) => *n,
        Some(Value::Int(n)) => *n as u64,
        other => panic!("unexpected active_connections {other:?}"),
    }
}

async fn await_connections(
    control: &QuackClient,
    port: u16,
    expected: u64,
) -> Result<Duration, u64> {
    let started = Instant::now();
    let mut last = active_connections(control, port).await;
    while started.elapsed() < Duration::from_secs(10) {
        if last == expected {
            return Ok(started.elapsed());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        last = active_connections(control, port).await;
    }
    Err(last)
}

#[tokio::test]
async fn seam_3_concurrency_and_pool() {
    let Some(uri) = server_uri() else { return };
    if !e2e_enabled() {
        return;
    }
    let _guard = LOCK.lock().await;
    init_tracing();
    let control = control_client(&uri).await;
    ensure_fact(
        &QuackPool::connect(
            &uri,
            QuackClientOptions {
                auth_token: token(),
                ..Default::default()
            },
            Default::default(),
        )
        .await
        .unwrap(),
    )
    .await;
    let nested = start_nested(&control, NESTED_CONCURRENCY_PORT).await;

    println!("\n===== SEAM 3: concurrency & pool (max_connections=4, 16 concurrent queries) =====");
    let pool = connect(&nested, 4).await;
    let factory = QuackTableFactory::new(Arc::clone(&pool));
    let ctx = session();
    ctx.register_table("fact", factory.table_provider("fact").await.unwrap())
        .unwrap();
    println!(
        "sessions after pool + provider creation: {}",
        active_connections(&control, NESTED_CONCURRENCY_PORT).await
    );

    // How many scans does one query open?
    for sql in [
        "SELECT * FROM fact LIMIT 10",
        "SELECT dim_id, sum(amount) FROM fact GROUP BY dim_id",
        "SELECT f.id FROM fact f JOIN fact g ON f.id = g.id WHERE f.id < 100",
    ] {
        let before = counts();
        let _ = df_collect(&ctx, sql).await.expect(sql);
        let d = delta(before);
        let plan = explain(&ctx, sql).await;
        let parts = plan
            .lines()
            .filter(|l| l.contains("VirtualExecutionPlan"))
            .count();
        println!(
            "scans per query: {sql}\n  prepares={} fetches={} VirtualExecutionPlan nodes={} target_partitions={}",
            d.prepares,
            d.fetches,
            parts,
            ctx.state().config().target_partitions()
        );
    }

    // 16 concurrent queries while sampling the server's session count.
    let ctx = Arc::new(ctx);
    let mut set = JoinSet::new();
    let started = Instant::now();
    for i in 0..16u32 {
        let ctx = Arc::clone(&ctx);
        set.spawn(async move {
            let sql = if i % 2 == 0 {
                format!("SELECT count(*), sum(amount) FROM fact WHERE dim_id = {i}")
            } else {
                format!("SELECT id, label FROM fact WHERE id % 1000 = {i} AND id < 2000000")
            };
            let t = Instant::now();
            let r = df_collect(&ctx, &sql).await.map(|b| batch_rows(&b));
            (i, r, t.elapsed())
        });
    }
    let mut max_seen = 0;
    let mut samples = 0;
    let mut results = Vec::new();
    loop {
        tokio::select! {
            joined = set.join_next() => match joined {
                Some(r) => results.push(r.unwrap()),
                None => break,
            },
            _ = tokio::time::sleep(Duration::from_millis(20)) => {
                let n = active_connections(&control, NESTED_CONCURRENCY_PORT).await;
                max_seen = max_seen.max(n);
                samples += 1;
            }
        }
    }
    let total = started.elapsed();
    let failures: Vec<_> = results.iter().filter(|(_, r, _)| r.is_err()).collect();
    println!(
        "16 concurrent queries: total wall={total:?} ok={} failed={} max sessions seen={max_seen} (samples={samples}) final sessions={}",
        results.len() - failures.len(),
        failures.len(),
        active_connections(&control, NESTED_CONCURRENCY_PORT).await
    );
    for (i, r, t) in &results {
        match r {
            Ok(n) => println!("  q{i}: {n} rows in {t:?}"),
            Err(e) => println!("  q{i}: ERROR {e}"),
        }
    }
    assert!(max_seen <= 4, "pool exceeded max_connections: {max_seen}");

    drop(ctx);
    drop(factory);
    let disconnects_before = DISCONNECTS.load(Ordering::Relaxed);
    drop(pool);
    match await_connections(&control, NESTED_CONCURRENCY_PORT, 0).await {
        Ok(t) => println!(
            "sessions after pool drop: 0 (settled in {t:?}, {} drop-disconnects logged)",
            DISCONNECTS.load(Ordering::Relaxed) - disconnects_before
        ),
        Err(n) => println!("sessions after pool drop: STILL {n} after 10s"),
    }
    let _ = stop_nested(&control, NESTED_CONCURRENCY_PORT).await;
}

// ===========================================================================
// SEAM 4: cancellation & abandonment (E2E)
// ===========================================================================

#[tokio::test]
async fn seam_4_cancellation() {
    let Some(uri) = server_uri() else { return };
    if !e2e_enabled() {
        return;
    }
    let _guard = LOCK.lock().await;
    init_tracing();
    let control = control_client(&uri).await;
    ensure_fact(
        &QuackPool::connect(
            &uri,
            QuackClientOptions {
                auth_token: token(),
                ..Default::default()
            },
            Default::default(),
        )
        .await
        .unwrap(),
    )
    .await;
    let nested = start_nested(&control, NESTED_CANCEL_PORT).await;
    let pool = connect(&nested, 2).await;
    let factory = QuackTableFactory::new(Arc::clone(&pool));
    let ctx = session();
    ctx.register_table("fact", factory.table_provider("fact").await.unwrap())
        .unwrap();

    println!("\n===== SEAM 4: cancellation & abandonment (max_connections=2) =====");
    let before = counts();
    let t = Instant::now();
    let out = df_rows(&ctx, "SELECT * FROM fact LIMIT 10").await;
    let d = delta(before);
    println!(
        "LIMIT 10 pushed: wall={:?} prepares={} fetches={} rows={}\n{}",
        t.elapsed(),
        d.prepares,
        d.fetches,
        out.lines().count().saturating_sub(4),
        explain(&ctx, "SELECT * FROM fact LIMIT 10").await
    );

    // DataFusion stops consuming: take one batch of a full scan and drop it.
    let before = counts();
    let t = Instant::now();
    let df = ctx.sql("SELECT * FROM fact").await.unwrap();
    let mut stream = df.execute_stream().await.unwrap();
    let first = stream.next().await.unwrap().unwrap();
    println!(
        "abandoned full scan: first batch {} rows after {:?}; sessions while held={}",
        first.num_rows(),
        t.elapsed(),
        active_connections(&control, NESTED_CANCEL_PORT).await
    );
    drop(stream);
    tokio::time::sleep(Duration::from_millis(200)).await;
    let d = delta(before);
    println!(
        "after drop: prepares={} fetches={} sessions={}\n  last PREPAREs: {:?}",
        d.prepares,
        d.fetches,
        active_connections(&control, NESTED_CANCEL_PORT).await,
        PREPARE_SQL
            .lock()
            .map(|l| l.iter().rev().take(d.prepares).cloned().collect::<Vec<_>>())
            .unwrap_or_default()
    );

    // A DataFusion-side limit after a step the unparser cannot push: a
    // window function over the whole table, then take 5.
    let before = counts();
    let t = Instant::now();
    let df = ctx
        .sql("SELECT id, row_number() OVER (ORDER BY amount) AS rn FROM fact")
        .await
        .unwrap()
        .limit(0, Some(5))
        .unwrap();
    let plan = df
        .clone()
        .explain(false, false)
        .unwrap()
        .collect()
        .await
        .unwrap();
    let out = df
        .collect()
        .await
        .map(|b| fmt(&b))
        .unwrap_or_else(|e| format!("ERROR {e}"));
    let d = delta(before);
    println!(
        "window + DataFrame.limit(5): wall={:?} prepares={} fetches={}\n{}\n{}",
        t.elapsed(),
        d.prepares,
        d.fetches,
        fmt(&plan)
            .lines()
            .filter(|l| l.contains("Exec") || l.contains("Virtual"))
            .collect::<Vec<_>>()
            .join("\n"),
        out
    );

    // Both pool slots must be usable again: two concurrent queries.
    let ctx = Arc::new(ctx);
    let a = df_collect(&ctx, "SELECT count(*) FROM fact WHERE dim_id = 1");
    let b = df_collect(&ctx, "SELECT count(*) FROM fact WHERE dim_id = 2");
    let (a, b) = tokio::join!(a, b);
    println!(
        "next two concurrent queries after abandonment: {} / {}; sessions={}",
        a.map(|_| "ok").unwrap_or("ERROR"),
        b.map(|_| "ok").unwrap_or("ERROR"),
        active_connections(&control, NESTED_CANCEL_PORT).await
    );
    drop(ctx);
    drop(factory);
    drop(pool);
    match await_connections(&control, NESTED_CANCEL_PORT, 0).await {
        Ok(t) => println!("sessions after pool drop: 0 (settled in {t:?})"),
        Err(n) => println!("sessions after pool drop: STILL {n} after 10s (LEAK)"),
    }
    let _ = stop_nested(&control, NESTED_CANCEL_PORT).await;
}

// ===========================================================================
// SEAM 5: failure modes
// ===========================================================================

#[tokio::test]
async fn seam_5_failure_modes() {
    let Some(uri) = server_uri() else { return };
    let _guard = LOCK.lock().await;
    init_tracing();
    let control = control_client(&uri).await;
    let main_pool = QuackPool::connect(
        &uri,
        QuackClientOptions {
            auth_token: token(),
            ..Default::default()
        },
        Default::default(),
    )
    .await
    .unwrap();
    let slow = e2e_enabled();
    if slow {
        ensure_fact(&main_pool).await;
    } else {
        main_pool.execute("CREATE OR REPLACE TABLE small AS SELECT range AS id, ('s' || range) AS label FROM range(100000)", None).await.unwrap();
    }
    let table = if slow { "fact" } else { "small" };

    println!("\n===== SEAM 5: failure modes =====");

    // (a) wrong token, at pool creation and via CREATE EXTERNAL TABLE.
    if token().is_some() {
        let mut p = params(&uri, 1, &[]);
        p.insert("auth_token".into(), SecretString::from("wrong"));
        match QuackConnectionPool::new(p).await {
            Ok(_) => println!("(a) wrong token: pool creation UNEXPECTEDLY succeeded"),
            Err(e) => println!("(a) wrong token, pool creation error:\n    {e}"),
        }
        let ctx = session();
        ctx.state_ref()
            .write()
            .table_factories_mut()
            .insert("QUACK".into(), Arc::new(QuackTableProviderFactory::new()));
        let out = df_rows(
            &ctx,
            &format!("CREATE EXTERNAL TABLE t STORED AS quack LOCATION '{table}' OPTIONS ('quack.uri' '{uri}', 'quack.auth_token' 'wrong')"),
        )
        .await;
        println!("(a) wrong token via CREATE EXTERNAL TABLE:\n    {out}");
    } else {
        println!("(a) wrong token: skipped, server has no token");
    }

    // (b) unreachable at pool creation; unreachable at query time.
    let mut p = params("quack:127.0.0.1:1", 1, &[]);
    p.insert("timeout".into(), SecretString::from("2"));
    match QuackConnectionPool::new(p).await {
        Ok(_) => println!("(b) unreachable: pool creation UNEXPECTEDLY succeeded"),
        Err(e) => println!("(b) unreachable at pool creation:\n    {e}"),
    }
    let nested = start_nested(&control, NESTED_UNREACHABLE_PORT).await;
    let pool = connect(&nested, 2).await;
    let ctx = session();
    ctx.register_table(
        "t",
        QuackTableFactory::new(Arc::clone(&pool))
            .table_provider(table)
            .await
            .unwrap(),
    )
    .unwrap();
    stop_nested(&control, NESTED_UNREACHABLE_PORT)
        .await
        .unwrap();
    let out = df_rows(&ctx, "SELECT count(*) FROM t").await;
    println!("(b) server stopped after provider creation, query error:\n    {out}");
    drop(ctx);
    drop(pool);

    // (c) restart mid-life: query, stop, re-serve, query again.
    let nested = start_nested(&control, NESTED_RESTART_PORT).await;
    let pool = connect(&nested, 2).await;
    let ctx = session();
    ctx.register_table(
        "t",
        QuackTableFactory::new(Arc::clone(&pool))
            .table_provider(table)
            .await
            .unwrap(),
    )
    .unwrap();
    let first = df_rows(&ctx, "SELECT count(*) FROM t").await;
    stop_nested(&control, NESTED_RESTART_PORT).await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let _ = start_nested(&control, NESTED_RESTART_PORT).await;
    let second = df_rows(&ctx, "SELECT count(*) FROM t").await;
    let third = df_rows(&ctx, "SELECT count(*) FROM t").await;
    println!(
        "(c) restart: before={} | first after restart={} | second after restart={}",
        first.lines().nth(3).unwrap_or("?").trim(),
        second.lines().nth(3).unwrap_or(&second).trim(),
        third.lines().nth(3).unwrap_or(&third).trim()
    );
    if second.starts_with("ERROR") {
        println!("    first-after-restart error text:\n    {second}");
    }
    drop(ctx);
    drop(pool);
    let _ = stop_nested(&control, NESTED_RESTART_PORT).await;

    // (d) mid-stream error vs PREPARE-time error. DataFusion must be able to
    // plan the statement, so the failing expression is a CAST over data
    // rather than DuckDB's `error()` (unknown to DataFusion's planner).
    let pool = connect(&uri, 2).await;
    let quack = pool.pool();
    let n_rows: i64 = if slow { 5_000_000 } else { 100_000 };
    quack
        .execute(
            &format!(
                "CREATE OR REPLACE TABLE e2e_midstream AS SELECT range AS id, \
                 CASE WHEN range < {} THEN CAST(range AS VARCHAR) ELSE 'boom' END AS s \
                 FROM range({n_rows})",
                n_rows - 1000
            ),
            None,
        )
        .await
        .unwrap();
    let ctx = session();
    let f = QuackTableFactory::new(Arc::clone(&pool));
    ctx.register_table("t", f.table_provider(table).await.unwrap())
        .unwrap();
    ctx.register_table("ms", f.table_provider("e2e_midstream").await.unwrap())
        .unwrap();
    for (name, sql) in [
        (
            "cast: every row fails",
            "SELECT CAST(label AS INTEGER) AS v FROM t",
        ),
        (
            "cast: only the last 1000 rows fail (1/3)",
            "SELECT CAST(s AS INTEGER) AS v FROM ms",
        ),
        (
            "cast: only the last 1000 rows fail (2/3)",
            "SELECT CAST(s AS INTEGER) AS v FROM ms",
        ),
        (
            "cast: only the last 1000 rows fail (3/3)",
            "SELECT CAST(s AS INTEGER) AS v FROM ms",
        ),
        (
            "cast: last rows fail, ORDER BY id (materialised)",
            "SELECT CAST(s AS INTEGER) AS v FROM ms ORDER BY id",
        ),
        (
            "cast: last rows fail, DataFusion-side filter after fetch",
            "SELECT v FROM (SELECT id, CAST(s AS INTEGER) AS v FROM ms) WHERE v % 7 = 0",
        ),
    ] {
        let before = counts();
        let df = match ctx.sql(sql).await {
            Ok(df) => df,
            Err(e) => {
                println!("(d) {name}: PLAN-TIME error: {e}");
                continue;
            }
        };
        let mut rows = 0usize;
        let mut batches = 0usize;
        let mut err = None;
        let t = Instant::now();
        match df.execute_stream().await {
            Err(e) => err = Some(format!("at execute(): {e}")),
            Ok(mut stream) => {
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(b) => {
                            rows += b.num_rows();
                            batches += 1;
                        }
                        Err(e) => {
                            err = Some(format!("after {batches} batches / {rows} rows: {e}"));
                            break;
                        }
                    }
                }
            }
        }
        let d = delta(before);
        println!(
            "(d) {name}: wall={:?} prepares_ok={} fetches_ok={} -> {}",
            t.elapsed(),
            d.prepares,
            d.fetches,
            err.unwrap_or_else(|| format!("NO ERROR, {rows} rows delivered (TRUNCATED RESULT?)"))
        );
    }
    let after = df_rows(&ctx, "SELECT count(*) FROM t").await;
    println!(
        "(d) pool still usable afterwards: {}",
        after.lines().nth(3).unwrap_or(&after).trim()
    );
    drop(ctx);
    drop(pool);

    // (e) request timeout of 1s on a slow, fully pushed-down query (own
    // nested server so a still-running server-side query cannot block the
    // rest of the suite). DataFusion plans the self cross-join; federation
    // sends it to DuckDB whole.
    main_pool
        .execute(
            "CREATE OR REPLACE TABLE e2e_slow AS SELECT range AS id FROM range(60000)",
            None,
        )
        .await
        .unwrap();
    let nested = start_nested(&control, NESTED_TIMEOUT_PORT).await;
    let pool = Arc::new(
        QuackConnectionPool::new(params(&nested, 1, &[("timeout", "1")]))
            .await
            .unwrap(),
    );
    let ctx = session();
    ctx.register_table(
        "s",
        QuackTableFactory::new(Arc::clone(&pool))
            .table_provider("e2e_slow")
            .await
            .unwrap(),
    )
    .unwrap();
    let slow_sql = "SELECT count(*) FROM s a JOIN s b ON a.id + b.id = -1";
    println!("(e) plan:\n{}", explain(&ctx, slow_sql).await);
    let t = Instant::now();
    let out = df_rows(&ctx, slow_sql).await;
    println!(
        "(e) timeout=1s slow query returned after {:?}:\n    {out}",
        t.elapsed()
    );
    let t = Instant::now();
    let out = df_rows(&ctx, "SELECT count(*) FROM s").await;
    println!(
        "(e) next query on the same pool after the timeout ({:?}): {} ; nested sessions={}",
        t.elapsed(),
        out.lines().nth(3).unwrap_or(&out).trim(),
        active_connections(&control, NESTED_TIMEOUT_PORT).await
    );
    drop(ctx);
    drop(pool);
    let _ = stop_nested(&control, NESTED_TIMEOUT_PORT).await;
}

// ===========================================================================
// PERFORMANCE (E2E)
// ===========================================================================

/// quack_protocol opens a new TCP connection per request
/// (`pool_max_idle_per_host(0)`), so a 5M-row scan leaves ~200 sockets in
/// TIME_WAIT (30s on macOS). Back-to-back scan runs exhaust the ~16k ephemeral
/// ports and fail with EADDRNOTAVAIL, so wait for the count to drop before
/// a scan-heavy section. Only implemented for macOS `netstat`; elsewhere a
/// fixed pause is used.
async fn drain_time_wait(uri: &str, label: &str) {
    let port = uri.rsplit(':').next().unwrap_or("0").to_string();
    let count = || -> Option<usize> {
        if !cfg!(target_os = "macos") {
            return None;
        }
        let out = Command::new("netstat")
            .args(["-an", "-p", "tcp"])
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let needle = format!(".{port} ");
        Some(
            text.lines()
                .filter(|l| l.contains(&needle) && l.contains("TIME_WAIT"))
                .count(),
        )
    };
    let started = Instant::now();
    let initial = count();
    let mut last = initial;
    while started.elapsed() < Duration::from_secs(45) {
        match last {
            Some(n) if n < 500 => break,
            None if started.elapsed() > Duration::from_secs(31) => break,
            _ => {}
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
        last = count();
    }
    println!(
        "  [{label}] TIME_WAIT sockets toward :{port}: {} -> {} after {:?}",
        initial.map_or("n/a".into(), |n| n.to_string()),
        last.map_or("n/a".into(), |n| n.to_string()),
        started.elapsed()
    );
}

async fn timed<F, Fut>(label: &str, runs: usize, mut f: F) -> (f64, usize, usize, Counts)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = (usize, usize)>,
{
    let mut secs = Vec::new();
    let mut rows = 0;
    let mut bytes = 0;
    let mut last = Counts {
        prepares: 0,
        fetches: 0,
    };
    for i in 0..runs {
        let before = counts();
        let t = Instant::now();
        let (r, b) = f().await;
        let s = t.elapsed().as_secs_f64();
        last = delta(before);
        rows = r;
        bytes = b;
        secs.push(s);
        println!(
            "  {label} run {}: {:.3}s rows={r} {:.1} Mrows/s {:.1} MB/s prepares={} fetches={} peak_rss={:.0}MB",
            i + 1,
            s,
            r as f64 / s / 1e6,
            b as f64 / s / 1e6,
            last.prepares,
            last.fetches,
            peak_rss_mb()
        );
    }
    let m = median(secs);
    println!(
        "  {label} MEDIAN: {:.3}s  {:.2} Mrows/s  {:.1} MB/s  ({} bytes of batches, {:.2} ms per fetch)",
        m,
        rows as f64 / m / 1e6,
        bytes as f64 / m / 1e6,
        bytes,
        if last.fetches > 0 { m * 1000.0 / last.fetches as f64 } else { 0.0 }
    );
    (m, rows, bytes, last)
}

#[tokio::test]
async fn perf_scan_and_aggregate() {
    let Some(uri) = server_uri() else { return };
    if !e2e_enabled() {
        return;
    }
    let _guard = LOCK.lock().await;
    init_tracing();
    let pool = connect(&uri, 8).await;
    print_env(&pool);
    let quack = pool.pool();
    ensure_fact(&quack).await;
    let ctx = session();
    ctx.register_table(
        "fact",
        QuackTableFactory::new(Arc::clone(&pool))
            .table_provider("fact")
            .await
            .unwrap(),
    )
    .unwrap();

    drain_time_wait(&uri, "before perf").await;
    println!("\n===== PERF: full scan of 5M rows (3 runs, median) =====");
    let (df_m, ..) = timed("DataFusion SELECT * FROM fact", 3, || {
        let ctx = &ctx;
        async move {
            let b = df_collect(ctx, "SELECT * FROM fact").await.expect("scan");
            (batch_rows(&b), batch_bytes(&b))
        }
    })
    .await;
    let (raw_m, ..) = timed("raw QuackPool into_record_batches", 3, || {
        let quack = &quack;
        async move {
            let b = raw_batches(quack, "SELECT * FROM fact")
                .await
                .expect("scan");
            (batch_rows(&b), batch_bytes(&b))
        }
    })
    .await;
    println!(
        "  provider overhead over raw transport: {:.1}%",
        (df_m / raw_m - 1.0) * 100.0
    );
    if duckdb_cli_version().is_some() {
        let mut secs = Vec::new();
        for _ in 0..3 {
            let (_, t) = duckdb_cli("SELECT 1").unwrap();
            secs.push(t.as_secs_f64());
        }
        let startup = median(secs);
        let mut secs = Vec::new();
        let mut out = String::new();
        for _ in 0..3 {
            let (o, t) =
                duckdb_cli("SELECT count(*), sum(id), sum(amount), max(label), max(d) FROM fact")
                    .unwrap();
            out = o;
            secs.push(t.as_secs_f64() - startup);
        }
        println!(
            "  local duckdb CLI, full-column aggregate over fact (materialisation ceiling, minus {:.0}ms process start-up): median {:.3}s -> {}",
            startup * 1000.0,
            median(secs),
            out.trim()
        );
        let mut secs = Vec::new();
        for _ in 0..3 {
            let (_, t) =
                duckdb_cli("COPY (SELECT * FROM fact) TO '/dev/null' (FORMAT CSV)").unwrap();
            secs.push(t.as_secs_f64() - startup);
        }
        println!(
            "  local duckdb CLI, COPY fact TO /dev/null (CSV): median {:.3}s",
            median(secs)
        );
    }

    println!("\n===== PERF: pushed-down aggregate =====");
    let agg = "SELECT dim_id, sum(amount) FROM fact GROUP BY 1";
    println!("plan:\n{}", explain(&ctx, agg).await);
    timed("DataFusion GROUP BY", 3, || {
        let ctx = &ctx;
        async move {
            let b = df_collect(ctx, agg).await.expect("agg");
            (batch_rows(&b), batch_bytes(&b))
        }
    })
    .await;
    timed("raw QuackPool GROUP BY", 3, || {
        let quack = &quack;
        async move {
            let b = raw_batches(quack, agg).await.expect("agg");
            (batch_rows(&b), batch_bytes(&b))
        }
    })
    .await;
    if duckdb_cli_version().is_some() {
        let mut secs = Vec::new();
        for _ in 0..3 {
            let (_, t) = duckdb_cli(agg).unwrap();
            secs.push(t.as_secs_f64());
        }
        println!(
            "  local duckdb CLI GROUP BY (incl. process start-up): median {:.3}s",
            median(secs)
        );
    }

    println!("\n===== PERF: per-request latency (SELECT 1 x 20) =====");
    let mut lat = Vec::new();
    for _ in 0..20 {
        let t = Instant::now();
        let _ = quack.values("SELECT 1").await.unwrap();
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    println!(
        "  PREPARE round trip median {:.2} ms (min {:.2}, max {:.2})",
        median(lat.clone()),
        lat.iter().cloned().fold(f64::MAX, f64::min),
        lat.iter().cloned().fold(0.0, f64::max)
    );

    println!("\n===== PERF: concurrent full scans (max_connections=8) =====");
    let ctx = Arc::new(ctx);
    for (label, via_df) in [("DataFusion", true), ("raw QuackPool", false)] {
        drain_time_wait(&uri, label).await;
        for n in [1usize, 4, 8] {
            let mut secs = Vec::new();
            let mut cpus = Vec::new();
            let mut rows_total = 0usize;
            for _ in 0..3 {
                let mut set = JoinSet::new();
                let t = Instant::now();
                let cpu0 = cpu_seconds();
                for _ in 0..n {
                    let ctx = Arc::clone(&ctx);
                    let quack = quack.clone();
                    set.spawn(async move {
                        if via_df {
                            batch_rows(&df_collect(&ctx, "SELECT * FROM fact").await.expect("scan"))
                        } else {
                            batch_rows(
                                &raw_batches(&quack, "SELECT * FROM fact")
                                    .await
                                    .expect("scan"),
                            )
                        }
                    });
                }
                rows_total = 0;
                while let Some(r) = set.join_next().await {
                    rows_total += r.unwrap();
                }
                secs.push(t.elapsed().as_secs_f64());
                cpus.push(cpu_seconds() - cpu0);
            }
            let m = median(secs);
            let c = median(cpus);
            println!(
                "  {label}: {n} concurrent scans: median wall {:.3}s aggregate {:.2} Mrows/s ({:.2} Mrows/s per scan); client CPU {:.2}s = {:.2} cores busy",
                m,
                rows_total as f64 / m / 1e6,
                rows_total as f64 / n as f64 / m / 1e6,
                c,
                c / m
            );
        }
    }
    println!("  peak RSS at end: {:.0} MB", peak_rss_mb());
    pool.close().await.ok();
}
