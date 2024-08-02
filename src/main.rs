use {rand::prelude::*, std::ops::Add};

#[derive(Debug, Clone)]
struct Matrix {
  rows: usize,
  columns: usize,
  inner: Vec<f64>,
}

impl Matrix {
  fn new(rows: usize, columns: usize) -> Self {
    let mut rng = thread_rng();
    Self {
      rows,
      columns,
      inner: (0..rows * columns)
        .map(|_| rng.gen_range(-1.0..=1.0))
        .collect(),
    }
  }

  fn multiply(&self, other: &Matrix) -> Matrix {
    debug_assert_eq!(
      self.columns, other.rows,
      "Invalid matrix dimensions for multiplication: {}x{} * {}x{}",
      self.rows, self.columns, other.rows, other.columns
    );

    let mut result = Matrix::new(self.rows, other.columns);

    for i in 0..self.rows {
      for j in 0..other.columns {
        let mut sum = 0.0;

        for k in 0..self.columns {
          sum += self.inner[i * self.columns + k]
            * other.inner[k * other.columns + j];
        }

        result.inner[i * result.columns + j] = sum;
      }
    }

    result
  }

  fn subtract(&self, other: &Matrix) -> Matrix {
    debug_assert_eq!(
      self.rows, other.rows,
      "Matrix rows must match for subtraction"
    );

    debug_assert_eq!(
      self.columns, other.columns,
      "Matrix columns must match for subtraction"
    );

    let mut result = Matrix::new(self.rows, self.columns);

    for i in 0..self.inner.len() {
      result.inner[i] = self.inner[i] - other.inner[i];
    }

    result
  }

  fn hadamard_product(&self, other: &Matrix) -> Matrix {
    debug_assert_eq!(
      self.rows, other.rows,
      "Matrix rows must match for Hadamard product"
    );

    debug_assert_eq!(
      self.columns, other.columns,
      "Matrix columns must match for Hadamard product"
    );

    let mut result = Matrix::new(self.rows, self.columns);

    for i in 0..self.inner.len() {
      result.inner[i] = self.inner[i] * other.inner[i];
    }

    result
  }

  fn apply<F>(&self, f: F) -> Matrix
  where
    F: Fn(f64) -> f64,
  {
    let mut result = self.clone();

    for i in 0..self.inner.len() {
      result.inner[i] = f(self.inner[i]);
    }

    result
  }

  fn transpose(&self) -> Matrix {
    let mut result = Matrix::new(self.columns, self.rows);

    for i in 0..self.rows {
      for j in 0..self.columns {
        result.inner[j * self.rows + i] = self.inner[i * self.columns + j];
      }
    }

    result
  }
}

impl Add for Matrix {
  type Output = Matrix;

  fn add(self, other: Matrix) -> Matrix {
    debug_assert_eq!(
      self.rows, other.rows,
      "Matrix rows must match for addition"
    );

    debug_assert_eq!(
      self.columns, other.columns,
      "Matrix columns must match for addition"
    );

    let mut result = Matrix::new(self.rows, self.columns);

    for i in 0..self.inner.len() {
      result.inner[i] = self.inner[i] + other.inner[i];
    }

    result
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
      hidden: 4,
      output: 3,
      learning_rate: 0.1,
      weight_input_hidden: Matrix::new(4, 3),
      weight_hidden_output: Matrix::new(3, 4),
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

  fn forward(&self, input: &Matrix) -> Matrix {
    let hidden = self
      .config
      .weight_input_hidden
      .multiply(input)
      .apply(sigmoid);

    self
      .config
      .weight_hidden_output
      .multiply(&hidden)
      .apply(sigmoid)
  }

  fn train(&mut self, input: &Matrix, target: &Matrix) {
    let hidden = self
      .config
      .weight_input_hidden
      .multiply(input)
      .apply(sigmoid);

    let output = self
      .config
      .weight_hidden_output
      .multiply(&hidden)
      .apply(sigmoid);

    let output_error = target.subtract(&output);

    let output_delta =
      output_error.hadamard_product(&output.apply(sigmoid_derivative));

    let hidden_error = self
      .config
      .weight_hidden_output
      .transpose()
      .multiply(&output_delta);

    let hidden_delta =
      hidden_error.hadamard_product(&hidden.apply(sigmoid_derivative));

    let hidden_transpose = hidden.transpose();

    let input_transpose = input.transpose();

    let weight_hidden_output_delta = output_delta
      .multiply(&hidden_transpose)
      .apply(|x| x * self.config.learning_rate);

    self.config.weight_hidden_output = self
      .config
      .weight_hidden_output
      .clone()
      .add(weight_hidden_output_delta);

    let weight_input_hidden_delta = hidden_delta
      .multiply(&input_transpose)
      .apply(|x| x * self.config.learning_rate);

    self.config.weight_input_hidden = self
      .config
      .weight_input_hidden
      .clone()
      .add(weight_input_hidden_delta);
  }
}

fn sigmoid(x: f64) -> f64 {
  1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
  x * (1.0 - x)
}

fn main() {
  let mut network = Network::new(NetworkConfig::default());

  let input = Matrix {
    rows: 3,
    columns: 1,
    inner: vec![0.5, 0.1, 0.9],
  };
  let target = Matrix {
    rows: 3,
    columns: 1,
    inner: vec![0.8, 0.2, 0.7],
  };

  println!("Input shape: {}x{}", input.rows, input.columns);

  println!("Target shape: {}x{}", target.rows, target.columns);

  println!(
    "Weight input hidden shape: {}x{}",
    network.config.weight_input_hidden.rows,
    network.config.weight_input_hidden.columns
  );

  println!(
    "Weight hidden output shape: {}x{}",
    network.config.weight_hidden_output.rows,
    network.config.weight_hidden_output.columns
  );

  for i in 0..10000 {
    network.train(&input, &target);

    if i % 1000 == 0 {
      let output = network.forward(&input);
      println!("Iteration {}: Output = {:?}", i, output);
    }
  }

  let output = network.forward(&input);

  println!("Final prediction:");
  println!("Input: {:?}", input);
  println!("Output: {:?}", output);
  println!("Target: {:?}", target);
}

#[cfg(test)]
mod tests {
  use {super::*, approx::assert_relative_eq};

  #[test]
  fn matrix_new() {
    let matrix = Matrix::new(2, 3);

    assert_eq!(matrix.rows, 2);
    assert_eq!(matrix.columns, 3);
    assert_eq!(matrix.inner.len(), 6);
  }

  #[test]
  fn matrix_multiply() {
    let a = Matrix {
      rows: 2,
      columns: 3,
      inner: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    };

    let b = Matrix {
      rows: 3,
      columns: 2,
      inner: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    };

    let result = a.multiply(&b);

    assert_eq!(result.rows, 2);
    assert_eq!(result.columns, 2);
    assert_eq!(result.inner, vec![58.0, 64.0, 139.0, 154.0]);
  }

  #[test]
  fn matrix_subtract() {
    let a = Matrix {
      rows: 2,
      columns: 2,
      inner: vec![1.0, 2.0, 3.0, 4.0],
    };

    let b = Matrix {
      rows: 2,
      columns: 2,
      inner: vec![0.5, 1.0, 1.5, 2.0],
    };

    let result = a.subtract(&b);

    assert_eq!(result.inner, vec![0.5, 1.0, 1.5, 2.0]);
  }

  #[test]
  fn matrix_hadamard_product() {
    let a = Matrix {
      rows: 2,
      columns: 2,
      inner: vec![1.0, 2.0, 3.0, 4.0],
    };

    let b = Matrix {
      rows: 2,
      columns: 2,
      inner: vec![2.0, 3.0, 4.0, 5.0],
    };

    let result = a.hadamard_product(&b);

    assert_eq!(result.inner, vec![2.0, 6.0, 12.0, 20.0]);
  }

  #[test]
  fn matrix_apply() {
    let a = Matrix {
      rows: 2,
      columns: 2,
      inner: vec![1.0, 2.0, 3.0, 4.0],
    };

    let result = a.apply(|x| x * 2.0);

    assert_eq!(result.inner, vec![2.0, 4.0, 6.0, 8.0]);
  }

  #[test]
  fn matrix_transpose() {
    let a = Matrix {
      rows: 2,
      columns: 3,
      inner: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    };

    let result = a.transpose();

    assert_eq!(result.rows, 3);
    assert_eq!(result.columns, 2);
    assert_eq!(result.inner, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
  }

  #[test]
  fn matrix_add() {
    let a = Matrix {
      rows: 2,
      columns: 2,
      inner: vec![1.0, 2.0, 3.0, 4.0],
    };

    let b = Matrix {
      rows: 2,
      columns: 2,
      inner: vec![5.0, 6.0, 7.0, 8.0],
    };

    let result = a.add(b);

    assert_eq!(result.inner, vec![6.0, 8.0, 10.0, 12.0]);
  }

  #[test]
  fn sigmoid_works() {
    assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-6);
    assert_relative_eq!(sigmoid(1.0), 0.7310585786300049, epsilon = 1e-6);
    assert_relative_eq!(sigmoid(-1.0), 0.2689414213699951, epsilon = 1e-6);
  }

  #[test]
  fn sigmoid_derivative_works() {
    assert_relative_eq!(sigmoid_derivative(0.5), 0.25, epsilon = 1e-6);

    assert_relative_eq!(
      sigmoid_derivative(0.7310585786300049),
      0.19661193324148185,
      epsilon = 1e-6
    );
  }

  #[test]
  fn network_forward() {
    let config = NetworkConfig::default();

    let network = Network::new(config);

    let input = Matrix {
      rows: 3,
      columns: 1,
      inner: vec![0.5, 0.1, 0.9],
    };

    let output = network.forward(&input);

    assert_eq!(output.rows, 3);
    assert_eq!(output.columns, 1);

    // We can't assert exact values due to random initialization,
    // but we can check that outputs are within the sigmoid range (0 to 1)
    for &value in output.inner.iter() {
      assert!(value >= 0.0 && value <= 1.0);
    }
  }

  #[test]
  fn network_train() {
    let config = NetworkConfig::default();

    let mut network = Network::new(config);

    let input = Matrix {
      rows: 3,
      columns: 1,
      inner: vec![0.5, 0.1, 0.9],
    };

    let target = Matrix {
      rows: 3,
      columns: 1,
      inner: vec![0.8, 0.2, 0.7],
    };

    let initial_output = network.forward(&input);

    // Train for a few iterations
    for _ in 0..100 {
      network.train(&input, &target);
    }

    let final_output = network.forward(&input);

    // Check that the output has moved closer to the target
    for i in 0..3 {
      assert!(
        (final_output.inner[i] - target.inner[i]).abs()
          < (initial_output.inner[i] - target.inner[i]).abs()
      );
    }
  }
}
