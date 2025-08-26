load './build/release/ml.duckdb_extension';


CREATE OR REPLACE MACRO scaling_params(table_name, column_list) AS TABLE
    FROM query_table(table_name)
    SELECT
        "avg_\0": avg(columns(column_list)),
        "std_\0": stddev_pop(columns(column_list)),
        "min_\0": min(columns(column_list)),
        "max_\0": max(columns(column_list));

create table autompg as from 'test/auto_mpg.csv';

CREATE OR REPLACE MACRO min_max_scaler(val, min_val, max_val) AS
	(val - min_val) / nullif(max_val - min_val, 0)
;

select ml_create('autompg', '
{"layers": [
    {"in": 5, "out": 64, "activation": "relu"},
    {"in": 64, "out": 32, "activation": "relu"},
    {"in": 32, "out": 1}
]}');

with
params as (
	from scaling_params('autompg', ['cylinders','displacement','horsepower','weight','acceleration','mpg'])
),
scaled as (
	select
		cylinders: min_max_scaler(
			cylinders,
			min_cylinders,
			max_cylinders
		),
		displacement: min_max_scaler(
			displacement,
			min_displacement,
			max_displacement
		),
		horsepower: min_max_scaler(
			horsepower,
			min_horsepower,
			max_horsepower
		),
		weight: min_max_scaler(
			weight,
			min_weight,
			max_weight
		),
		weight: min_max_scaler(
			weight,
			min_weight,
			max_weight
		),
		acceleration: min_max_scaler(
			acceleration,
			min_acceleration,
			max_acceleration
		),
		mpg: min_max_scaler(
			mpg,
			min_mpg,
			max_mpg
		),
	from autompg
	join params on true
)
select
	*,
	mpg_pred: ml_train('autompg', [cylinders::float, displacement::float, horsepower::float, weight::float, acceleration::float], [mpg::float])[1],
from scaled;
