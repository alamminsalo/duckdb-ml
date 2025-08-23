use burn::{data::dataloader::batcher::Batcher, prelude::*};
use serde::{Deserialize, Serialize};

// XYValue holds input features -> targets data in X -> Y.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XYValue(pub Vec<f64>, pub Vec<f64>);

#[derive(Clone, Debug)]
pub struct TensorBatch<B: Backend, const D: usize> {
    pub features: Tensor<B, D>,
    pub targets: Tensor<B, D>,
}

pub struct TensorBatcher;

impl<B: Backend, const D: usize> Batcher<B, XYValue, TensorBatch<B, D>> for TensorBatcher {
    fn batch(&self, items: Vec<XYValue>, device: &B::Device) -> TensorBatch<B, D> {
        let mut features: Vec<Tensor<B, 1>> = Vec::new();
        let mut targets: Vec<Tensor<B, 1>> = Vec::new();

        items.into_iter().for_each(|item| {
            features.push(Tensor::<B, 1>::from_floats(item.0.as_slice(), device));
            targets.push(Tensor::<B, 1>::from_floats(item.1.as_slice(), device));
        });

        TensorBatch {
            features: Tensor::stack(features, 0),
            targets: Tensor::stack(targets, 0),
        }
    }
}
