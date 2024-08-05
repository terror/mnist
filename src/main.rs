use {
  ndarray::{Array, Array2, ArrayView1, ArrayView2, Axis},
  ndarray_rand::rand_distr::Uniform,
  ndarray_rand::RandomExt,
  rand::seq::SliceRandom,
  rayon::prelude::*,
  std::{
    fs::File,
    io::{self, Read},
    path::Path,
    process,
    sync::{Arc, RwLock},
  },
};

type Result<T = (), E = anyhow::Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub struct MnistData {
  training_images: Arc<Array2<f64>>,
  training_labels: Arc<Array2<f64>>,
  test_images: Arc<Array2<f64>>,
  test_labels: Arc<Array2<f64>>,
}

impl MnistData {
  fn load(path: &str) -> Result<MnistData> {
    Ok(MnistData {
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

  fn read_images<P: AsRef<Path>>(path: P) -> io::Result<Array2<f64>> {
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 4];

    file.read_exact(&mut buffer)?;
    let magic_number = u32::from_be_bytes(buffer);

    if magic_number != 2051 {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid image file format",
      ));
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

  fn read_labels<P: AsRef<Path>>(path: P) -> io::Result<Array2<f64>> {
    let mut file = File::open(path)?;
    let mut buffer = [0u8; 4];

    file.read_exact(&mut buffer)?;
    let magic_number = u32::from_be_bytes(buffer);

    if magic_number != 2049 {
      return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "Invalid label file format",
      ));
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

#[derive(Debug)]
struct NetworkConfig {
  learning_rate: f64,
  weight_input_hidden: Array2<f64>,
  weight_hidden_output: Array2<f64>,
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

fn run() -> Result {
  let mnist_data = MnistData::load("data")?;

  println!(
    "Loaded {} training images",
    mnist_data.training_images.nrows()
  );

  println!(
    "Loaded {} training labels",
    mnist_data.training_labels.nrows()
  );

  println!("Loaded {} test images", mnist_data.test_images.nrows());

  println!("Loaded {} test labels", mnist_data.test_labels.nrows());

  let network = Arc::new(RwLock::new(Network::new(NetworkConfig::default())));

  let (epochs, batch_size) = (10, 128);

  let mut indices: Vec<usize> =
    (0..mnist_data.training_images.nrows()).collect();

  for epoch in 0..epochs {
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

    println!(
      "Epoch {}: Test accuracy = {:.2}%",
      epoch + 1,
      accuracy * 100.0
    );
  }

  Ok(())
}

fn main() {
  if let Err(error) = run() {
    eprintln!("error: {error}");
    process::exit(1);
  }
}

#[cfg(test)]
mod tests {
  use {super::*, approx::assert_relative_eq, ndarray::array};

  #[test]
  fn mnist_data_load() {
    let mnist_data = MnistData::load("data").unwrap();

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
}
