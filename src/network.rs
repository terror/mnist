use super::*;

#[derive(Clone, Debug)]
pub(crate) struct Network {
  config: NetworkConfig,
}

impl Network {
  pub(crate) fn new(config: NetworkConfig) -> Self {
    Self { config }
  }

  pub(crate) fn forward(&self, input: ArrayView2<f64>) -> Array2<f64> {
    let hidden = self.config.weight_input_hidden.dot(&input.t()).mapv(relu);
    self.config.weight_hidden_output.dot(&hidden).mapv(sigmoid)
  }

  pub(crate) fn train_batch(
    &mut self,
    inputs: ArrayView2<f64>,
    targets: ArrayView2<f64>,
  ) {
    let batch_size = inputs.nrows();

    let hidden = self.config.weight_input_hidden.dot(&inputs.t()).mapv(relu);
    let output = self.config.weight_hidden_output.dot(&hidden).mapv(sigmoid);

    let output_error = &targets.t() - &output;
    let output_delta = &output_error * &output.mapv(sigmoid_derivative);

    let hidden_error = self.config.weight_hidden_output.t().dot(&output_delta);
    let hidden_delta = &hidden_error * &hidden.mapv(relu_derivative);

    let weight_hidden_output_delta = output_delta.dot(&hidden.t());
    let weight_input_hidden_delta = hidden_delta.dot(&inputs);

    self.config.weight_hidden_output += &(weight_hidden_output_delta
      * (self.config.learning_rate / batch_size as f64));

    self.config.weight_input_hidden += &(weight_input_hidden_delta
      * (self.config.learning_rate / batch_size as f64));
  }

  pub(crate) fn evaluate(
    &self,
    inputs: ArrayView2<f64>,
    targets: ArrayView2<f64>,
  ) -> f64 {
    let outputs = self.forward(inputs);

    let predicted = outputs.map_axis(Axis(0), |row| argmax(&row.view()));

    let actual = targets.map_axis(Axis(1), |row| argmax(&row.view()));

    (predicted
      .iter()
      .zip(actual.iter())
      .filter(|&(a, b)| a == b)
      .count() as f64)
      / inputs.nrows() as f64
  }

  pub(crate) fn save_weights(&self, path: &PathBuf) -> Result {
    let serializable_config: SerializableNetworkConfig =
      self.config.clone().into();

    let file = File::create(path).context("failed to create weights file")?;

    serde_json::to_writer(file, &serializable_config)
      .context("failed to serialize network weights")?;

    Ok(())
  }

  pub(crate) fn load_weights(path: &PathBuf) -> Result<Self> {
    let file = File::open(path).context("failed to open weights file")?;

    let serializable_config: SerializableNetworkConfig =
      serde_json::from_reader(file)
        .context("failed to deserialize network weights")?;

    let config = NetworkConfig::try_from(serializable_config)?;

    Ok(Self::new(config))
  }
}

#[cfg(test)]
mod tests {
  use {
    super::*,
    approx::assert_relative_eq,
    ndarray::{array, Array},
    tempdir::TempDir,
  };

  fn compare_arrays_with_tolerance(
    a: &Array2<f64>,
    b: &Array2<f64>,
    tolerance: f64,
  ) -> bool {
    if a.shape() != b.shape() {
      return false;
    }

    a.iter()
      .zip(b.iter())
      .all(|(&x, &y)| (x - y).abs() < tolerance)
  }

  #[test]
  fn network_forward() {
    let config = NetworkConfig {
      learning_rate: 0.1,
      weight_input_hidden: array![[0.1, 0.2], [0.3, 0.4]],
      weight_hidden_output: array![[0.5, 0.6], [0.7, 0.8]],
    };

    let network = Network::new(config);

    let input = array![[1.0, 1.0]];
    let output = network.forward(input.view());

    let expected_output = array![
      [sigmoid(0.5 * 0.3 + 0.6 * 0.7)],
      [sigmoid(0.7 * 0.3 + 0.8 * 0.7)]
    ];

    assert_eq!(
      output.shape(),
      expected_output.shape(),
      "Output shape mismatch"
    );

    assert_relative_eq!(
      output[[0, 0]],
      expected_output[[0, 0]],
      epsilon = 1e-6
    );

    assert_relative_eq!(
      output[[1, 0]],
      expected_output[[1, 0]],
      epsilon = 1e-6
    );
  }

  #[test]
  fn network_train_batch() {
    let config = NetworkConfig {
      learning_rate: 0.1,
      weight_input_hidden: Array::random((5, 3), Uniform::new(-0.1, 0.1)),
      weight_hidden_output: Array::random((2, 5), Uniform::new(-0.1, 0.1)),
    };

    let mut network = Network::new(config);

    let inputs = array![[1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]];
    let targets = array![[1.0, 0.0], [0.0, 1.0]];

    network.train_batch(inputs.view(), targets.view());

    assert!(
      network.config.weight_input_hidden.iter().any(|&x| x != 0.0),
      "Input-hidden weights were not updated"
    );

    assert!(
      network
        .config
        .weight_hidden_output
        .iter()
        .any(|&x| x != 0.0),
      "Hidden-output weights were not updated"
    );
  }

  #[test]
  fn network_evaluate() {
    let config = NetworkConfig {
      learning_rate: 0.1,
      weight_input_hidden: array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
      weight_hidden_output: array![[0.7, 0.8], [0.9, 1.0]],
    };

    let network = Network::new(config);

    let inputs = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
    let targets = array![[1.0, 0.0], [0.0, 1.0]];

    let accuracy = network.evaluate(inputs.view(), targets.view());

    assert!(accuracy >= 0.0 && accuracy <= 1.0);
  }

  #[test]
  fn network_save_and_load_weights() {
    let dir = TempDir::new("test").unwrap();

    let weight_path = dir.path().join("test_weights.json");

    let original_config = NetworkConfig::default();
    let original_network = Network::new(original_config);

    original_network.save_weights(&weight_path).unwrap();

    let loaded_network = Network::load_weights(&weight_path).unwrap();

    assert_relative_eq!(
      original_network.config.learning_rate,
      loaded_network.config.learning_rate
    );

    assert!(compare_arrays_with_tolerance(
      &original_network.config.weight_input_hidden,
      &loaded_network.config.weight_input_hidden,
      1e-6
    ));

    assert!(compare_arrays_with_tolerance(
      &original_network.config.weight_hidden_output,
      &loaded_network.config.weight_hidden_output,
      1e-6
    ));
  }
}
