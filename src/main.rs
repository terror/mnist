use rand::prelude::*;

#[derive(Debug)]
struct Matrix {
  rows: usize,
  columns: usize,
  inner: Vec<f64>,
}

impl Matrix {
  fn new(rows: usize, columns: usize) -> Self {
    Self {
      rows,
      columns,
      inner: (0..rows * columns)
        .map(|_| thread_rng().gen_range(-1.0..=1.0))
        .collect(),
    }
  }

  fn multiply(self, matrix: Matrix) -> Self {
    todo!()
  }
}

#[derive(Debug)]
struct NetworkConfig {
  input: usize,
  hidden: usize,
  output: usize,
  learning_rate: f64,
  weight_input_hidden: Matrix,
  weight_hidden_output: Matrix,
}

impl Default for NetworkConfig {
  fn default() -> Self {
    Self {
      input: 3,
      hidden: 3,
      output: 3,
      learning_rate: 0.3,
      weight_input_hidden: Matrix::new(3, 3),
      weight_hidden_output: Matrix::new(3, 3),
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
}

fn main() {
  let network = Network::new(NetworkConfig::default());
  println!("{:?}", network);
}
