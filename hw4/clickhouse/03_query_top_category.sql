-- Категория наибольшей транзакции по каждому штату
USE fraud_demo;

SELECT
    us_state,
    argMax(cat_id, amount) AS category_with_max_amount,
    max(amount) AS max_amount
FROM teta_transactions_raw
GROUP BY us_state
ORDER BY us_state;
