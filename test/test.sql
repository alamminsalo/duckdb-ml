load './build/debug/extension/ml/ml.duckdb_extension';

select ml_createmodel('foo', '{"layers": []}');
from ml_listmodels();
