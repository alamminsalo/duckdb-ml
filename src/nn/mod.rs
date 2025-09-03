mod batcher;
mod model;
mod train;

use batcher::XYValue;
use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    optim::AdamConfig,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
};
use chrono::{DateTime, Utc};
use model::*;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
pub use train::TrainingConfig;

type B = Autodiff<NdArray<f32>>;

static MODEL_REGISTRY: OnceLock<Mutex<HashMap<String, model::Model<B>>>> = OnceLock::new();

pub fn build_model(name: &str, spec_json: &str) -> Result<Model<B>, Box<dyn std::error::Error>> {
    Ok(Model::<B>::from_spec(name, spec_json, &Default::default())?)
}

pub fn register_model(model: Model<B>) -> Result<(), Box<dyn std::error::Error>> {
    put_model(model)?;
    Ok(())
}

pub fn list_models() -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    let list = Vec::from_iter(
        MODEL_REGISTRY
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()?
            .iter()
            .map(|(_, v)| (v.name.clone(), v.specjson.clone())),
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

pub fn put_model(model: Model<B>) -> Result<(), Box<dyn std::error::Error>> {
    MODEL_REGISTRY
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()?
        .insert(model.name.clone(), model);

    Ok(())
}

pub fn predict(
    name: &str,
    features: Vec<Vec<f32>>,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    // Get model without trainable weights
    let model = get_model(name)?.valid();

    // Run inference
    let pred = model.forward(make_tensor2d(&features));

    // Collect into vec format
    Ok(pred
        .split(1, 0)
        .into_iter()
        .map(|t| t.to_data().into_vec().unwrap())
        .collect())
}

pub fn train_model_reg<B: AutodiffBackend>(
    model: Model<B>,
    features: Vec<Vec<f32>>,
    targets: Vec<Vec<f32>>,
    config: &TrainingConfig,
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
    let mut test_xy = vec![];

    // Train-test split with static 30%
    // TODO: parametrize
    let split_frac = 0.3;
    let split_at = (train_xy.len() as f32 * split_frac) as usize;
    if split_at > 0 {
        test_xy = train_xy.split_off(split_at);
    }

    // Add this line in your Rust code to include an ISO timestamp variable with the current time
    let timestamp: DateTime<Utc> = Utc::now();
    let artifact_path = format!(
        "models/{}/{}",
        &model.name,
        timestamp.format("%Y%m%d_%H%M%S")
    );

    train::train_reg(
        model,
        train_xy,
        test_xy,
        config,
        AdamConfig::new().init(),
        &artifact_path,
        &Default::default(),
    )
}

fn make_tensor2d<B: Backend>(input: &Vec<Vec<f32>>) -> Tensor<B, 2> {
    let mut output: Vec<Tensor<B, 1>> = Vec::new();
    let device = Default::default();

    input.into_iter().for_each(|item| {
        output.push(Tensor::<B, 1>::from_floats(item.as_slice(), &device));
    });

    Tensor::stack(output, 0)
}
