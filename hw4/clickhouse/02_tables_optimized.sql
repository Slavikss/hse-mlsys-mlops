-- Оптимизированная схема хранения: LowCardinality + компактные типы + партиционирование
CREATE DATABASE IF NOT EXISTS fraud_demo;
USE fraud_demo;

DROP VIEW IF EXISTS mv_teta_transactions;
DROP TABLE IF EXISTS teta_transactions_raw;
DROP TABLE IF EXISTS teta_transactions_kafka;

CREATE TABLE teta_transactions_kafka
(
    transaction_time String,
    merch LowCardinality(String),
    cat_id LowCardinality(String),
    amount Decimal(12, 2),
    name_1 String,
    name_2 String,
    gender LowCardinality(String),
    street String CODEC(ZSTD(3)),
    one_city LowCardinality(String),
    us_state LowCardinality(FixedString(2)),
    post_code String,
    lat Decimal(10, 6),
    lon Decimal(10, 6),
    population_city UInt32,
    jobs LowCardinality(String),
    merchant_lat Decimal(10, 6),
    merchant_lon Decimal(10, 6)
) ENGINE = Kafka
SETTINGS
    kafka_broker_list = 'kafka:9092',
    kafka_topic_list = 'teta_transactions',
    kafka_group_name = 'ck_teta_ingest_v3',
    kafka_format = 'CSV',
    kafka_row_delimiter = '\n';

CREATE TABLE teta_transactions_raw
(
    transaction_time DateTime,
    merch LowCardinality(String),
    cat_id LowCardinality(String),
    amount Decimal(12, 2),
    name_1 String,
    name_2 String,
    gender LowCardinality(String),
    street String CODEC(ZSTD(3)),
    one_city LowCardinality(String),
    us_state LowCardinality(FixedString(2)),
    post_code String,
    lat Decimal(10, 6),
    lon Decimal(10, 6),
    population_city UInt32,
    jobs LowCardinality(String),
    merchant_lat Decimal(10, 6),
    merchant_lon Decimal(10, 6),
    INDEX idx_amount amount TYPE minmax GRANULARITY 1
) ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(transaction_time)
ORDER BY (us_state, cat_id, transaction_time)
TTL transaction_time + INTERVAL 10 YEAR DELETE
SETTINGS index_granularity = 8192, storage_policy = 'default';

CREATE MATERIALIZED VIEW mv_teta_transactions
TO teta_transactions_raw
AS
SELECT
    parseDateTimeBestEffort(transaction_time) AS transaction_time,
    merch,
    cat_id,
    amount,
    name_1,
    name_2,
    gender,
    street,
    one_city,
    us_state,
    post_code,
    lat,
    lon,
    population_city,
    jobs,
    merchant_lat,
    merchant_lon
FROM teta_transactions_kafka;
