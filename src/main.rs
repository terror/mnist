use {
  anyhow::{bail, Context, Result},
  clap::Parser,
  indicatif::{ProgressBar, ProgressStyle},
  ndarray::{Array, Array2, ArrayView1, ArrayView2, Axis},
  ndarray_rand::rand_distr::Uniform,
  ndarray_rand::RandomExt,
  rand::seq::SliceRandom,
  rayon::prelude::*,
  serde::{Deserialize, Serialize},
  std::{
    fs::{read, File},
    io::Read,
    path::Path,
    path::PathBuf,
    process,
    sync::{Arc, RwLock},
  },
};

#[derive(Debug, Parser)]
struct Arguments {
  #[clap(subcommand)]
  subcommand: Subcommand,
}

impl Arguments {
  fn run(self) -> Result<()> {
    self.subcommand.run()
  }
}

#[derive(Debug, Parser)]
enum Subcommand {
  Train {
    #[clap(short, long, default_value = "50")]
    epochs: usize,
    #[clap(short, long, default_value = "128")]
    batch_size: usize,
    #[clap(short, long, default_value = "weights.json")]
    output: PathBuf,
  },
  Predict {
    #[clap(short, long)]
    weights: PathBuf,
    #[clap(short, long)]
    image: PathBuf,
  },
}

impl Subcommand {
  fn run(self) -> Result<()> {
    match self {
      Subcommand::Train {
        batch_size,
        epochs,
        output,
      } => train(epochs, batch_size, output),
      Subcommand::Predict { weights, image } => predict(weights, image),
    }
  }
}

#[derive(Debug)]
pub struct Dataset {
  training_images: Arc<Array2<f64>>,
  training_labels: Arc<Array2<f64>>,
  test_images: Arc<Array2<f64>>,
  test_labels: Arc<Array2<f64>>,
}

impl Dataset {
  fn load(path: &str) -> Result<Dataset> {
    Ok(Dataset {
      training_images: Arc::new(Self::read_images(
        Path::new(path).join("train-images-idx3-ubyte"),
      )?),
      training_labels: Arc::new(Self::read_labels(
        Path::new(path).join("train-labels-idx1-ubyte"),
      )?),
      test_images: Arc::new(Self::read_images(
        Path::new(path).join("t10k-images-idx3-ubyte"),
      )?),
      test_labels: Arc::new(Self::read_labels(
        Path::new(path).join("t10k-labels-idx1-ubyte"),
      )?),
    })
  }

  fn read_images<P: AsRef<Path>>(path: P) -> Result<Array2<f64>> {
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 4];

    file.read_exact(&mut buffer)?;
    let magic_number = u32::from_be_bytes(buffer);

    if magic_number != 2051 {
      bail!("invalid image file format");
    }

    file.read_exact(&mut buffer)?;
    let num_images = u32::from_be_bytes(buffer) as usize;

    file.read_exact(&mut buffer)?;
    let num_rows = u32::from_be_bytes(buffer) as usize;

    file.read_exact(&mut buffer)?;
    let num_cols = u32::from_be_bytes(buffer) as usize;

    let mut image_buffer = vec![0u8; num_images * num_rows * num_cols];
    file.read_exact(&mut image_buffer)?;

    let images = Array2::from_shape_vec(
      (num_images, num_rows * num_cols),
      image_buffer
        .into_iter()
        .map(|pixel| pixel as f64 / 255.0)
        .collect(),
    )
    .unwrap();

    Ok(images)
  }

  fn read_labels<P: AsRef<Path>>(path: P) -> Result<Array2<f64>> {
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 4];

    file.read_exact(&mut buffer)?;
    let magic_number = u32::from_be_bytes(buffer);

    if magic_number != 2049 {
      bail!("invalid label file format");
    }

    file.read_exact(&mut buffer)?;
    let num_labels = u32::from_be_bytes(buffer) as usize;

    let mut label_buffer = vec![0u8; num_labels];
    file.read_exact(&mut label_buffer)?;

    let mut labels = Array2::zeros((num_labels, 10));

    for (i, &label) in label_buffer.iter().enumerate() {
      labels[[i, label as usize]] = 1.0;
    }

    Ok(labels)
  }
}

#[derive(Debug, Serialize, Deserialize)]
struct SerializableNetworkConfig {
  learning_rate: f64,
  weight_input_hidden: Vec<f64>,
  weight_hidden_output: Vec<f64>,
  input_hidden_shape: (usize, usize),
  hidden_output_shape: (usize, usize),
}

#[derive(Debug)]
struct NetworkConfig {
  learning_rate: f64,
  weight_input_hidden: Array2<f64>,
  weight_hidden_output: Array2<f64>,
}

impl NetworkConfig {
  fn to_serializable(&self) -> SerializableNetworkConfig {
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

  fn from_serializable(config: SerializableNetworkConfig) -> Result<Self> {
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

impl Default for NetworkConfig {
  fn default() -> Self {
    Self {
      learning_rate: 0.1,
      weight_input_hidden: Array::random((128, 784), Uniform::new(-0.1, 0.1)),
      weight_hidden_output: Array::random((10, 128), Uniform::new(-0.1, 0.1)),
    }
  }
}

#[derive(Debug)]
struct Network {
  config: NetworkConfig,
}

impl Network {
  fn new(config: NetworkConfig) -> Self {
    Self { config }
  }

  fn forward(&self, input: ArrayView2<f64>) -> Array2<f64> {
    let hidden = self.config.weight_input_hidden.dot(&input.t()).mapv(relu);
    self.config.weight_hidden_output.dot(&hidden).mapv(sigmoid)
  }

  fn train_batch(&mut self, inputs: ArrayView2<f64>, targets: ArrayView2<f64>) {
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

  fn evaluate(&self, inputs: ArrayView2<f64>, targets: ArrayView2<f64>) -> f64 {
    let outputs = self.forward(inputs);

    let predicted = outputs.map_axis(Axis(0), |row| argmax(&row));

    let actual = targets.map_axis(Axis(1), |row| argmax(&row));

    (predicted
      .iter()
      .zip(actual.iter())
      .filter(|&(a, b)| a == b)
      .count() as f64)
      / inputs.nrows() as f64
  }

  fn save_weights(&self, path: &PathBuf) -> Result<()> {
    let serializable_config = self.config.to_serializable();

    let file = File::create(path).context("Failed to create weights file")?;

    serde_json::to_writer(file, &serializable_config)
      .context("Failed to serialize network weights")?;

    Ok(())
  }

  fn load_weights(path: &PathBuf) -> Result<Self> {
    let file = File::open(path).context("Failed to open weights file")?;

    let serializable_config: SerializableNetworkConfig =
      serde_json::from_reader(file)
        .context("Failed to deserialize network weights")?;

    let config = NetworkConfig::from_serializable(serializable_config)?;

    Ok(Self::new(config))
  }

  fn predict(&self, input: ArrayView2<f64>) -> Array2<f64> {
    self.forward(input)
  }
}

fn sigmoid(x: f64) -> f64 {
  1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
  x * (1.0 - x)
}

fn relu(x: f64) -> f64 {
  x.max(0.0)
}

fn relu_derivative(x: f64) -> f64 {
  if x > 0.0 {
    1.0
  } else {
    0.0
  }
}

fn argmax(row: &ArrayView1<f64>) -> usize {
  row
    .iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .map(|(index, _)| index)
    .unwrap()
}

fn train(epochs: usize, batch_size: usize, output: PathBuf) -> Result<()> {
  let mnist_data =
    Dataset::load("data").context("Failed to load MNIST dataset")?;

  println!("Dataset loaded successfully:");
  println!("  Training images: {}", mnist_data.training_images.nrows());
  println!("  Training labels: {}", mnist_data.training_labels.nrows());
  println!("  Test images: {}", mnist_data.test_images.nrows());
  println!("  Test labels: {}", mnist_data.test_labels.nrows());

  let network = Arc::new(RwLock::new(Network::new(NetworkConfig::default())));

  let mut indices: Vec<usize> =
    (0..mnist_data.training_images.nrows()).collect();

  let progress_bar = ProgressBar::new(epochs as u64);

  progress_bar.set_style(
    ProgressStyle::default_bar()
      .template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} Epochs {msg}",
      )?
      .progress_chars("=>-"),
  );

  for _ in 0..epochs {
    indices.shuffle(&mut rand::thread_rng());

    indices.par_chunks(batch_size).for_each(|batch| {
      let batch_inputs = mnist_data.training_images.select(Axis(0), batch);

      let batch_targets = mnist_data.training_labels.select(Axis(0), batch);

      network
        .write()
        .unwrap()
        .train_batch((&batch_inputs).into(), (&batch_targets).into());
    });

    let accuracy = network
      .read()
      .unwrap()
      .evaluate(mnist_data.test_images.view(), mnist_data.test_labels.view());

    progress_bar.set_message(format!("Accuracy: {:.2}%", accuracy * 100.0));

    progress_bar.inc(1);
  }

  progress_bar.finish_with_message("Training complete");

  network
    .read()
    .unwrap()
    .save_weights(&output)
    .context("Failed to save network weights")?;

  println!("Saved weights to {}", output.display());

  Ok(())
}

fn predict(weights: PathBuf, image_path: PathBuf) -> Result<()> {
  let network = Network::load_weights(&weights.clone())?;

  println!("Loaded weights from {}", weights.display());

  let image = read_image(image_path)?;
  let prediction = network.predict(image.view());
  let digit = argmax(&prediction.view().row(0));

  println!("Predicted digit: {}", digit);

  Ok(())
}

fn read_image(path: PathBuf) -> Result<Array2<f64>> {
  let image_data = read(path)?;

  let image: image::ImageBuffer<image::Luma<u8>, Vec<u8>> =
    image::load_from_memory(&image_data)?.to_luma8();

  let (width, height) = image.dimensions();

  if width != 28 || height != 28 {
    bail!("Image must be 28x28 pixels");
  }

  let flat_image: Vec<f64> = image
    .into_raw()
    .into_iter()
    .map(|p| p as f64 / 255.0)
    .collect();

  Ok(Array2::from_shape_vec((1, 784), flat_image)?)
}

fn main() {
  if let Err(error) = Arguments::parse().run() {
    eprintln!("error: {error}");
    process::exit(1);
  }
}

#[cfg(test)]
mod tests {
  use {
    super::*, approx::assert_relative_eq, ndarray::array, tempdir::TempDir,
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
  fn sigmoid_works() {
    assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-6);
    assert_relative_eq!(sigmoid(1.0), 0.7310585786300049, epsilon = 1e-6);
    assert_relative_eq!(sigmoid(-1.0), 0.2689414213699951, epsilon = 1e-6);
  }

  #[test]
  fn relu_works() {
    assert_eq!(relu(1.0), 1.0);
    assert_eq!(relu(-1.0), 0.0);
    assert_eq!(relu(0.0), 0.0);
  }

  #[test]
  fn argmax_works() {
    let arr = array![0.1, 0.3, 0.2, 0.4, 0.1];
    assert_eq!(argmax(&arr.view()), 3);
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
  fn mnist_data_load() {
    let mnist_data = Dataset::load("data").unwrap();

    assert_eq!(mnist_data.training_images.nrows(), 60000);
    assert_eq!(mnist_data.training_images.ncols(), 784);
    assert_eq!(mnist_data.training_labels.nrows(), 60000);
    assert_eq!(mnist_data.training_labels.ncols(), 10);
    assert_eq!(mnist_data.test_images.nrows(), 10000);
    assert_eq!(mnist_data.test_images.ncols(), 784);
    assert_eq!(mnist_data.test_labels.nrows(), 10000);
    assert_eq!(mnist_data.test_labels.ncols(), 10);
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
