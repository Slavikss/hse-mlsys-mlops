import logging
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from psycopg2 import OperationalError, sql

from streaming.config import AppConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
LOGGER = logging.getLogger("streamlit-ui")

cfg = AppConfig()
st.set_page_config(page_title="Fraud Monitoring", page_icon="🛡️", layout="wide")
st.title("📊 Мониторинг антифрода")


@st.cache_resource(show_spinner=False)
def get_db_connection():
    return psycopg2.connect(
        host=cfg.database.host,
        port=cfg.database.port,
        dbname=cfg.database.database,
        user=cfg.database.user,
        password=cfg.database.password,
    )


def fetch_dataframe(
    query: Union[sql.SQL, str],
    params: Optional[tuple] = None,
) -> pd.DataFrame:
    try:
        conn = get_db_connection()
        dataframe = pd.read_sql_query(
            query if isinstance(query, str) else query.as_string(conn),
            conn,
            params=params,
        )
        return dataframe
    except OperationalError:
        st.error(
            "Не удалось подключиться к базе данных. "
            "Убедитесь, что контейнер Postgres запущен."
        )
        raise
    except psycopg2.errors.UndefinedTable:
        st.warning("Таблица с предсказаниями пока не создана. Подождите генерацию данных.")
        return pd.DataFrame()


st.sidebar.header("Панель управления")
limit = st.sidebar.slider(
    "Сколько последних транзакций анализировать",
    min_value=50,
    max_value=500,
    value=150,
    step=50,
)

if st.sidebar.button("Обновить данные"):
    st.rerun()

summary_df = fetch_dataframe(
    "SELECT "
    "COUNT(*) AS total, "
    "SUM(CASE WHEN fraud_flag = 1 THEN 1 ELSE 0 END) AS frauds, "
    "AVG(score) AS avg_score, "
    "MAX(score) AS max_score, "
    "MIN(processed_at) AS first_seen, "
    "MAX(processed_at) AS last_seen "
    f"FROM {cfg.database.table_name}"
)

recent_df = fetch_dataframe(
    sql.SQL(
        "SELECT transaction_id, score, fraud_flag, processed_at "
        "FROM {table} "
        "ORDER BY processed_at DESC "
        "LIMIT %s"
    ).format(table=sql.Identifier(cfg.database.table_name)),
    params=(limit,),
)

fraud_df = fetch_dataframe(
    sql.SQL(
        "SELECT transaction_id, score, fraud_flag, processed_at "
        "FROM {table} "
        "WHERE fraud_flag = 1 "
        "ORDER BY processed_at DESC "
        "LIMIT 10"
    ).format(table=sql.Identifier(cfg.database.table_name))
)

if summary_df.empty or recent_df.empty:
    st.info(
        "Пока нет данных для отображения. Отправьте транзакции в Kafka, "
        "и результаты появятся автоматически."
    )
    st.stop()


summary = summary_df.iloc[0]
total = int(summary["total"])
frauds = int(summary["frauds"] or 0)
fraud_share = frauds / total if total else 0.0
avg_score = float(summary["avg_score"] or 0.0)
max_score = float(summary["max_score"] or 0.0)
last_seen = summary["last_seen"]

metrics_cols = st.columns(4)
metrics_cols[0].metric("Всего записей", f"{total:,}".replace(",", " "))
metrics_cols[1].metric(
    "Подозрительных",
    f"{frauds:,}".replace(",", " "),
    delta=f"{fraud_share * 100:.1f} %",
)
metrics_cols[2].metric("Средний скор", f"{avg_score:.3f}")
metrics_cols[3].metric(
    "Максимальный скор",
    f"{max_score:.3f}",
    delta="последние 24 ч" if total else None,
)

if pd.notna(last_seen):
    last_seen_ts = pd.to_datetime(last_seen)
    st.caption(f"Последнее обновление витрины: {last_seen_ts:%d.%m.%Y %H:%M:%S}")

recent_df["processed_at"] = pd.to_datetime(recent_df["processed_at"])
timeline_df = recent_df.sort_values("processed_at")

chart_col, hist_col = st.columns(2)
with chart_col:
    st.subheader("Динамика скоринга")
    st.line_chart(
        timeline_df.set_index("processed_at")["score"],
        height=280,
    )

with hist_col:
    st.subheader("Распределение скорингов")
    counts, bin_edges = np.histogram(recent_df["score"], bins=15, range=(0, 1))
    hist_df = pd.DataFrame(
        {
            "Интервал": [
                f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}"
                for i in range(len(bin_edges) - 1)
            ],
            "Количество": counts,
        }
    ).set_index("Интервал")
    st.bar_chart(hist_df, height=280)

st.subheader("Последние предсказания")
st.dataframe(
    recent_df.rename(
        columns={
            "transaction_id": "Transaction ID",
            "score": "Score",
            "fraud_flag": "Fraud",
            "processed_at": "Processed At",
        }
    ).style.format({"Score": "{:.4f}"}),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Фродовые события")
if fraud_df.empty:
    st.info("Пока нет транзакций с флагом фрода. Хороший знак!")
else:
    fraud_df["processed_at"] = pd.to_datetime(fraud_df["processed_at"])
    st.dataframe(
        fraud_df.rename(
            columns={
                "transaction_id": "Transaction ID",
                "score": "Score",
                "fraud_flag": "Fraud",
                "processed_at": "Processed At",
            }
        ).style.format({"Score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    flagged_trend = fraud_df.sort_values("processed_at")
    st.line_chart(
        flagged_trend.set_index("processed_at")["score"],
        height=220,
    )

st.caption(
    "Данные отображаются только из PostgreSQL. "
    "Kafka → inference → Postgres → Streamlit."
)
