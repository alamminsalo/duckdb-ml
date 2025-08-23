use crate::batcher::{TensorBatch, TensorBatcher, XYValue};
use crate::net::Net;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Dataset;
use burn::nn::loss::{CrossEntropyLoss, MseLoss, Reduction};
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::{
    config::Config,
    data::{
        dataloader::{DataLoader, DataLoaderBuilder},
        dataset::InMemDataset,
    },
    module::AutodiffModule,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::{backend::Backend, Tensor},
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};
use std::sync::Arc;

/// Training configuration
#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

/// Create artifact directory
fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Train an existing model with provided datasets
pub fn train<B: AutodiffBackend>(
    train_set: Vec<XYValue>,
    test_set: Vec<XYValue>,
    artifact_dir: &str,
    mut model: Net<B>,
    config: TrainingConfig,
    device: B::Device,
    mut optimizer: impl Optimizer<Net<B>, B>,
) {
    create_artifact_dir(artifact_dir);

    // Save config for reproducibility
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    // Set RNG seed
    B::seed(config.seed);

    // Build DataLoaders from provided tensors
    let dataloader_train: Arc<dyn DataLoader<B, TensorBatch<B, 2>>> =
        DataLoaderBuilder::new(TensorBatcher)
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .build(InMemDataset::new(train_set));

    //let dataloader_test: Arc<dyn DataLoader<B, TensorBatch<B, 2>>> =
    //DataLoaderBuilder::new(TensorBatcher).build(InMemDataset::new(test_set));

    // Simply make a static test batch
    let test_batch = {
        let tb = TensorBatcher;
        tb.batch(test_set, &device)
    };

    // Implement our training loop.
    for epoch in 1..config.num_epochs + 1 {
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.features);

            // Calculate loss
            let loss = MseLoss::new().forward(output, batch.targets, Reduction::Auto);

            // Gradients for the current backward pass
            let grads = loss.backward();

            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);

            // Update the model using the optimizer.
            model = optimizer.step(config.learning_rate, model, grads);

            // Get the model without autodiff.
            let test_loss = {
                let model_valid = model.valid();

                MseLoss::new().forward(
                    model_valid.forward(test_batch.features.clone()),
                    test_batch.targets.clone(),
                    Reduction::Auto,
                )
            };

            println!(
                "[Train - Epoch {} - Iteration {}] Loss Train {:.3} | Loss Test {:.3}",
                epoch,
                iteration,
                loss.into_scalar(),
                test_loss.into_scalar(),
            );
        }
    }
}
