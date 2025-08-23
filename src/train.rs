use crate::batcher::{TensorBatch, TensorBatcher, XYValue};
use crate::net::Net;
use burn::data::dataloader::batcher::Batcher;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::{
    config::Config,
    data::{
        dataloader::{DataLoader, DataLoaderBuilder},
        dataset::InMemDataset,
    },
    module::AutodiffModule,
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

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Train an existing model with provided datasets
pub fn train_reg<B: AutodiffBackend>(
    mut model: Net<B>,
    train_set: Vec<XYValue>,
    test_set: Vec<XYValue>,
    config: TrainingConfig,
    mut optimizer: impl Optimizer<Net<B>, B>,
    artifact_dir: &str,
    device: &B::Device,
) -> Net<B> {
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

    // Simply make a static test batch of full test dataset
    let test_batch = {
        let tb = TensorBatcher;
        tb.batch(test_set, device)
    };

    // Use simple mse loss
    // TODO: possibly parametrize other loss functions
    let lossfn = MseLoss::new();

    // Implement our training loop.
    for epoch in 1..config.num_epochs + 1 {
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.features);

            // Calculate loss
            let loss = lossfn.forward(output, batch.targets, Reduction::Mean);

            // Gradients for the current backward pass
            let grads = loss.backward();

            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);

            // Update the model using the optimizer.
            model = optimizer.step(config.learning_rate, model, grads);

            // Get the model without autodiff.
            let test_loss = {
                let model_valid = model.valid();

                lossfn.forward(
                    model_valid.forward(test_batch.features.clone()),
                    test_batch.targets.clone(),
                    Reduction::Mean,
                )
            };

            println!(
                "[Train: Epoch {} - Iteration {} | Loss: Train {:.3} - Test {:.3}]",
                epoch,
                iteration,
                loss.into_scalar(),
                test_loss.into_scalar(),
            );
        }
    }

    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batcher::XYValue;
    use crate::net::{Net, NetworkSpec};
    use burn::{
        backend::{Autodiff, NdArray},
        optim::AdamConfig,
        prelude::Tensor,
    };

    type B = Autodiff<NdArray<f32>>;

    #[test]
    fn test_train_simple() {
        // Inline JSON spec
        let json = r#"
        {
            "layers": [
                { "in": 4, "out": 8, "activation": "relu" },
                { "in": 8, "out": 1, "activation": "tanh" }
            ]
        }
        "#;

        let device = Default::default();

        // Parse the JSON into NetworkSpec
        let spec: NetworkSpec = serde_json::from_str(json).unwrap();
        let model = Net::<B>::from_spec(spec, &device);

        // Example input vec
        let train_set = vec![XYValue(vec![0., 0., 0., 0.], vec![0.])];
        let test_set = vec![XYValue(vec![0., 0., 0., 0.], vec![0.])];

        let model = train_reg(
            model,
            train_set,
            test_set,
            TrainingConfig::new(),
            AdamConfig::new().init(),
            "/tmp/__test_artifacts/test_train_simple",
            &device,
        );

        let input =
            Tensor::<B, 2>::from_floats([[0.5, -0.1, 0.3, 0.8], [1.0, 0.0, -0.2, 0.4]], &device);

        // Forward pass
        let output = model.forward(input);

        // Check output shape
        let data = output.to_data();
        assert_eq!(data.shape[0], 2, "Batch size should be 2");
        assert_eq!(data.shape[1], 1, "Output feature size should be 1");
    }
}
