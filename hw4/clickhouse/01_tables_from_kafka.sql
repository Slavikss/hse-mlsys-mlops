-- Базовый сценарий: читаем CSV с именами колонок из Kafka и складываем в MergeTree
CREATE DATABASE IF NOT EXISTS fraud_demo;
USE fraud_demo;

DROP VIEW IF EXISTS mv_teta_transactions;
DROP TABLE IF EXISTS teta_transactions_kafka;
DROP TABLE IF EXISTS teta_transactions_raw;

CREATE TABLE teta_transactions_kafka
(
    transaction_time String,
    merch String,
    cat_id String,
    amount Decimal(12, 2),
    name_1 String,
    name_2 String,
    gender String,
    street String,
    one_city String,
    us_state String,
    post_code String,
    lat Float64,
    lon Float64,
    population_city UInt32,
    jobs String,
    merchant_lat Float64,
    merchant_lon Float64
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
    merch String,
    cat_id String,
    amount Decimal(12, 2),
    name_1 String,
    name_2 String,
    gender String,
    street String,
    one_city String,
    us_state String,
    post_code String,
    lat Float64,
    lon Float64,
    population_city UInt32,
    jobs String,
    merchant_lat Float64,
    merchant_lon Float64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(transaction_time)
ORDER BY (us_state, transaction_time);

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
