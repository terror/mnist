use super::*;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SerializableNetworkConfig {
  learning_rate: f64,
  weight_input_hidden: Vec<f64>,
  weight_hidden_output: Vec<f64>,
  input_hidden_shape: (usize, usize),
  hidden_output_shape: (usize, usize),
}

#[derive(Clone, Debug)]
pub(crate) struct NetworkConfig {
  pub(crate) learning_rate: f64,
  pub(crate) weight_input_hidden: Array2<f64>,
  pub(crate) weight_hidden_output: Array2<f64>,
}

impl Default for NetworkConfig {
  fn default() -> Self {
    NetworkConfig {
      learning_rate: 0.1,
      weight_input_hidden: Array2::random((100, 784), Uniform::new(-0.1, 0.1)),
      weight_hidden_output: Array2::random((10, 100), Uniform::new(-0.1, 0.1)),
    }
  }
}

impl Into<SerializableNetworkConfig> for NetworkConfig {
  fn into(self) -> SerializableNetworkConfig {
    SerializableNetworkConfig {
      learning_rate: self.learning_rate,
      weight_input_hidden: self
        .weight_input_hidden
        .clone()
        .into_raw_vec_and_offset()
        .0,
      weight_hidden_output: self
        .weight_hidden_output
        .clone()
        .into_raw_vec_and_offset()
        .0,
      input_hidden_shape: self.weight_input_hidden.dim(),
      hidden_output_shape: self.weight_hidden_output.dim(),
    }
  }
}

impl TryFrom<SerializableNetworkConfig> for NetworkConfig {
  type Error = anyhow::Error;

  fn try_from(config: SerializableNetworkConfig) -> Result<Self> {
    Ok(Self {
      learning_rate: config.learning_rate,
      weight_input_hidden: Array2::from_shape_vec(
        config.input_hidden_shape,
        config.weight_input_hidden,
      )?,
      weight_hidden_output: Array2::from_shape_vec(
        config.hidden_output_shape,
        config.weight_hidden_output,
      )?,
    })
  }
}
