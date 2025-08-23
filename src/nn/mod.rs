mod batcher;
mod model;
mod train;

use burn::{backend::NdArray, tensor::Device};
use model::*;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

type B = NdArray<f32>;

static DEVICE: OnceLock<Device<B>> = OnceLock::new();
static MODEL_REGISTRY: OnceLock<Mutex<HashMap<String, model::Model<B>>>> = OnceLock::new();

pub fn register_model(name: &str, spec: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Parse the JSON into NetworkSpec
    let spec: NetworkSpec = serde_json::from_str(spec)?;

    //  Build model
    let device = DEVICE.get_or_init(|| Default::default());
    let model = Model::<B>::from_spec(spec, &device);

    put_model(name, model)?;

    Ok(())
}

pub fn list_models() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    Ok(MODEL_REGISTRY
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()?
        .keys()
        .cloned()
        .collect())
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
