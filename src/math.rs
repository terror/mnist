use super::*;

pub(crate) fn argmax<D>(x: &ArrayView<f64, D>) -> usize
where
  D: ndarray::Dimension,
{
  x.iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .map(|(index, _)| index)
    .unwrap()
}

pub(crate) fn relu(x: f64) -> f64 {
  x.max(0.0)
}

pub(crate) fn relu_derivative(x: f64) -> f64 {
  if x > 0.0 {
    1.0
  } else {
    0.0
  }
}

pub(crate) fn sigmoid(x: f64) -> f64 {
  1.0 / (1.0 + (-x).exp())
}

pub(crate) fn sigmoid_derivative(x: f64) -> f64 {
  x * (1.0 - x)
}

#[cfg(test)]
mod tests {
  use {super::*, approx::assert_relative_eq, ndarray::array};

  #[test]
  fn argmax_works() {
    let arr = array![0.1, 0.3, 0.2, 0.4, 0.1];
    assert_eq!(argmax(&arr.view()), 3);
  }

  #[test]
  fn relu_works() {
    assert_eq!(relu(1.0), 1.0);
    assert_eq!(relu(-1.0), 0.0);
    assert_eq!(relu(0.0), 0.0);
  }

  #[test]
  fn sigmoid_works() {
    assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-6);
    assert_relative_eq!(sigmoid(1.0), 0.7310585786300049, epsilon = 1e-6);
    assert_relative_eq!(sigmoid(-1.0), 0.2689414213699951, epsilon = 1e-6);
  }
}
