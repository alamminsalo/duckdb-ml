mod batcher;
mod model;
mod train;

use batcher::XYValue;
use burn::{
    backend::{Autodiff, NdArray},
    optim::AdamConfig,
    tensor::backend::AutodiffBackend,
    //tensor::Device,
};
use model::*;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

type B = Autodiff<NdArray<f32>>;

//static DEVICE: OnceLock<Device<B>> = OnceLock::new();
static MODEL_REGISTRY: OnceLock<Mutex<HashMap<String, model::Model<B>>>> = OnceLock::new();
static MODEL_SPECS: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();

pub fn register_model(name: &str, spec_json: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Parse the JSON into NetworkSpec
    let spec: NetworkSpec = serde_json::from_str(spec_json)?;

    //  Build model
    //let device = DEVICE.get_or_init(|| Default::default());
    let model = Model::<B>::from_spec(spec, &Default::default());

    put_model(name, model)?;

    let _ = MODEL_SPECS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()?
        .insert(name.to_string(), spec_json.to_string());

    Ok(())
}

pub fn list_models() -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    let list = Vec::from_iter(
        MODEL_SPECS
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()?
            .iter()
            .map(|(k, v)| (k.clone(), v.clone())),
    );

    Ok(list)
}

pub fn get_model(name: &str) -> Result<Model<B>, Box<dyn std::error::Error>> {
    Ok(MODEL_REGISTRY
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()?
        .get(name)
        .ok_or("Could not get model")?
        .clone())
}

pub fn put_model(name: &str, model: Model<B>) -> Result<(), Box<dyn std::error::Error>> {
    let _ = MODEL_REGISTRY
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()?
        .insert(name.to_string(), model);

    Ok(())
}

pub fn train_model_reg<B: AutodiffBackend>(
    model: Model<B>,
    features: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
) -> Model<B> {
    assert_eq!(
        features.len(),
        targets.len(),
        "Non-equal length inputs for training!"
    );

    let mut train_xy: Vec<XYValue> = features
        .into_iter()
        .zip(targets)
        .map(|(x, y)| XYValue(x, y))
        .collect();

    // Train-test split with static 30%
    // TODO: parametrize
    let split_frac = 0.3;
    let split_at = (train_xy.len() as f32 * split_frac) as usize;
    let test_xy = if split_at > 0 {
        train_xy.split_off(split_at)
    } else {
        vec![]
    };

    train::train_reg(
        model,
        train_xy,
        test_xy,
        train::TrainingConfig::new(),
        AdamConfig::new().init(),
        "/tmp/__test_artifacts/test_train_autompg",
        &Default::default(),
    )
}
