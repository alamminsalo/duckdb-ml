load './build/debug/extension/ml/ml.duckdb_extension';

select ml_createmodel('foo', '{"layers": [{"in": 2, "out": 8, "activation": "relu"}, {"in": 8, "out": 1}]}');
from ml_listmodels();

select ml_trainmodel('foo', [0.,0.], [0.]);
