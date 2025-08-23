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
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_fcn_forward_pass() {
        // Inline JSON spec
        let json = r#"
        {
            "layers": [
                { "in": 4, "out": 8, "activation": "relu", "batch_norm": true, "dropout": 0.1 },
                { "in": 8, "out": 3, "activation": "tanh" }
            ]
        }
        "#;

        // Parse the JSON into NetworkSpec
        let spec: NetworkSpec = serde_json::from_str(json).unwrap();

        // Create device and build model
        let device = Default::default();
        let model = Net::<B>::from_spec(spec, &device);

        // Example input: batch_size=2, features=4
        let input =
            Tensor::<B, 2>::from_floats([[0.5, -0.1, 0.3, 0.8], [1.0, 0.0, -0.2, 0.4]], &device);

        // Forward pass
        let output = model.forward(input);

        // Check output shape
        let data = output.to_data();
        assert_eq!(data.shape[0], 2, "Batch size should be 2");
        assert_eq!(data.shape[1], 3, "Output feature size should be 3");
    }

    #[test]
    fn test_fcn_no_activation() {
        // Simple network with no activations
        let json = r#"
        {
            "layers": [
                { "in": 3, "out": 3 }
            ]
        }
        "#;

        let spec: NetworkSpec = serde_json::from_str(json).unwrap();
        let device = Default::default();
        let model = Net::<B>::from_spec(spec, &device);

        let input = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let output = model.forward(input);
        let data = output.to_data();

        assert_eq!(data.shape[0], 1, "Batch size should be 1");
        assert_eq!(data.shape[1], 3, "Output feature size should be 3");
    }
}
