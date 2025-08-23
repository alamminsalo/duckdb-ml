use chrono::Local;
use duckdb::Value;
use std::fs;
use std::path::PathBuf;

pub fn model_dir(modelname: &str) -> PathBuf {
    let dir = std::path::Path::new("models").join(modelname);
    if !dir.exists() {
        fs::create_dir_all(&dir).expect("failed to create model directory");
    }
    dir
}

pub fn weight_file_path(modelname: &str) -> PathBuf {
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    model_dir(modelname).join(format!("weights_{}.pt", timestamp))
}

pub fn extract_2d_array(val: &Value) -> Result<Vec<Vec<f64>>, String> {
    match val {
        Value::List(list) => {
            let mut out = vec![];
            for row in list {
                match row {
                    Value::List(inner) => {
                        let mut inner_out = vec![];
                        for v in inner {
                            match v {
                                Value::Double(d) => inner_out.push(*d),
                                Value::Float(f) => inner_out.push(*f as f64),
                                Value::BigInt(i) => inner_out.push(*i as f64),
                                _ => return Err("X/y must be nested LIST<DOUBLE>".into()),
                            }
                        }
                        out.push(inner_out);
                    }
                    _ => return Err("X/y must be LIST<LIST<DOUBLE>>".into()),
                }
            }
            Ok(out)
        }
        _ => Err("Expected LIST<LIST<DOUBLE>>".into()),
    }
}
