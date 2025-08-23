use burn::{
    nn::{BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu, Tanh},
    prelude::*,
};
use serde::Deserialize;


/// One layer specification
#[derive(Deserialize, Debug)]
struct LayerSpec {
    #[serde(rename = "in")]
    in_features: usize,
    #[serde(rename = "out")]
    out_features: usize,
    activation: Option<String>, // "relu", "tanh", etc.
    batch_norm: Option<bool>,   // default = false
    dropout: Option<f64>,       // default = 0.0
}

/// Network specification
#[derive(Deserialize, Debug)]
struct NetworkSpec {
    layers: Vec<LayerSpec>,
    lossfn: String,
}

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    linears: Vec<Linear<B>>,
    batch_norms: Vec<Option<BatchNorm<B, 0>>>,
    dropouts: Vec<Option<Dropout>>,
    activations: Vec<Option<String>>,
}

impl<B: Backend> Net<B> {
    /// Build a model from JSON spec
    fn from_spec(spec: NetworkSpec, device: &B::Device) -> Self {
        let mut linears = Vec::new();
        let mut batch_norms = Vec::new();
        let mut dropouts = Vec::new();
        let mut activations = Vec::new();

        for l in spec.layers {
            // Linear layer
            linears.push(LinearConfig::new(l.in_features, l.out_features).init(device));

            // BatchNorm (optional)
            batch_norms.push(if l.batch_norm.unwrap_or(false) {
                Some(BatchNormConfig::new(l.out_features).init(device))
            } else {
                None
            });

            // Dropout (optional)
            dropouts.push(if let Some(rate) = l.dropout {
                if rate > 0.0 {
                    Some(DropoutConfig::new(rate).init())
                } else {
                    None
                }
            } else {
                None
            });

            // Activation (optional)
            activations.push(l.activation);
        }

        Self {
            linears,
            batch_norms,
            dropouts,
            activations,
        }
    }

    /// Forward pass
    pub fn forward(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        for ((lin, bn), (do_, act)) in self
            .linears
            .iter()
            .zip(self.batch_norms.iter())
            .zip(self.dropouts.iter().zip(self.activations.iter()))
        {
            x = lin.forward(x);

            if let Some(bn_layer) = bn {
                x = bn_layer.forward(x);
            }

            if let Some(drop_layer) = do_ {
                x = drop_layer.forward(x);
            }

            if let Some(a) = act {
                match a.as_str() {
                    "relu" => x = Relu::new().forward(x),
                    "tanh" => x = Tanh::new().forward(x),
                    _ => {}
                }
            }
        }
        x
    }

    //pub fn forward_reg(
    //    &self,
    //    features: Tensor<B, 2>,
    //    targets: Tensor<B, 2>,
    //) -> RegressionOutput<B> {
    //    let pred = self.forward(features);

    //    Regression
    //}
}

//impl<B: AutodiffBackend> TrainStep<TensorBatch<B, 2>, RegressionOutput<B>> for Net<B> {
//    fn step(&self, batch: TensorBatch<B, 2>) -> TrainOutput<RegressionOutput<B>> {
//        let item = self.forward_reg(batch.features, batch.targets);
//        TrainOutput::new(self, item.loss.backward(), item)
//    }
//}
//
//impl<B: Backend> ValidStep<TensorBatch<B, 2>, RegressionOutput<B>> for Net<B> {
//    fn step(&self, batch: TensorBatch<B, 2>) -> RegressionOutput<B> {
//        self.forward_reg(batch.features, batch.targets)
//    }
//}
