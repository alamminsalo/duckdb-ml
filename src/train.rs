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
    pub epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 0.01)]
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

    // Simply make a static test batch of full test dataset
    let (train_tensor, test_tensor) = {
        let tb = TensorBatcher;
        (
            tb.batch(train_set.clone(), device),
            tb.batch(test_set, device),
        )
    };

    // Build DataLoaders from provided tensors
    let dataloader_train: Arc<dyn DataLoader<B, TensorBatch<B, 2>>> =
        DataLoaderBuilder::new(TensorBatcher)
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .build(InMemDataset::new(train_set));

    // Use simple mse loss
    // TODO: possibly parametrize other loss functions
    let lossfn = MseLoss::new();

    // Implement our training loop.
    for epoch in 1..config.epochs + 1 {
        for (_, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.features);

            // Calculate loss
            let loss = lossfn.forward(output, batch.targets, Reduction::Mean);

            // Gradients for the current backward pass
            let grads = loss.backward();

            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);

            // Update the model using the optimizer.
            model = optimizer.step(config.learning_rate, model, grads);
        }

        // Print statistics at end of epoch
        // Get the model without autodiff and collect train/test losses.
        let model_valid = model.valid();

        let train_loss = lossfn.forward(
            model_valid.forward(train_tensor.features.clone()),
            train_tensor.targets.clone(),
            Reduction::Mean,
        );

        let test_loss = lossfn.forward(
            model_valid.forward(test_tensor.features.clone()),
            test_tensor.targets.clone(),
            Reduction::Mean,
        );

        println!(
            "[Train: Epoch={} LossTrain={:.3} LossTest={:.3}]",
            epoch,
            train_loss.into_scalar(),
            test_loss.into_scalar(),
        );
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

    use csv::ReaderBuilder;
    use std::error::Error;

    /// Loads a regression dataset from CSV into features and targets.
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file.
    ///
    /// # Returns
    /// * `(features, targets)` where:
    ///     - `features` is a Vec of rows, each row is Vec<f32>
    ///     - `targets` is a Vec<f32> of target values
    fn load_csv(path: &str) -> Result<(Vec<Vec<f32>>, Vec<f32>), Box<dyn Error>> {
        let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;

        let mut features: Vec<Vec<f32>> = Vec::new();
        let mut targets: Vec<f32> = Vec::new();

        for result in rdr.records() {
            let record = result?;

            // Last column is target
            let mut row: Vec<f32> = Vec::new();
            for i in 0..record.len() - 1 {
                row.push(record[i].parse::<f32>()?);
            }
            features.push(row);

            let target_val = record[record.len() - 1].parse::<f32>()?;
            targets.push(target_val);
        }

        Ok((features, targets))
    }

    #[test]
    fn test_train_simple() {
        // Inline JSON spec
        let json = r#"
        {
            "layers": [
                { "in": 4, "out": 8, "activation": "relu" },
                { "in": 8, "out": 1 }
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

    #[test]
    fn test_train_autompg() {
        // Inline JSON spec
        let json = r#"
        {
            "layers": [
                { "in": 5, "out": 64, "activation": "relu" },
                { "in": 64, "out": 64, "activation": "relu" },
                { "in": 64, "out": 1 }
            ]
        }
        "#;

        let device = Default::default();

        // Parse the JSON into NetworkSpec
        let spec: NetworkSpec = serde_json::from_str(json).unwrap();
        let model = Net::<B>::from_spec(spec, &device);

        // Example input vec
        let (features, targets) = load_csv("test/auto_mpg.csv").unwrap();

        let mut iter = features
            .into_iter()
            .zip(targets)
            .map(|(features, target)| XYValue(features, vec![target]));

        let test_set = iter.clone().take(50).collect();
        let train_set = iter.collect();

        let model = train_reg(
            model,
            train_set,
            test_set,
            TrainingConfig::new(),
            AdamConfig::new().init(),
            "/tmp/__test_artifacts/test_train_autompg",
            &device,
        );
    }
}
