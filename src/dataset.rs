use super::*;

#[derive(Debug)]
pub struct Dataset {
  pub(crate) training_images: Arc<Array2<f64>>,
  pub(crate) training_labels: Arc<Array2<f64>>,
  pub(crate) test_images: Arc<Array2<f64>>,
  pub(crate) test_labels: Arc<Array2<f64>>,
}

impl Dataset {
  pub(crate) fn load(path: &str) -> Result<Dataset> {
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

#[cfg(test)]
mod tests {
  use super::*;

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
}
