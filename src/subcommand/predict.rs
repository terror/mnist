use super::*;

#[derive(Debug, Parser)]
pub(crate) struct Predict {
  #[clap(short, long)]
  weights: PathBuf,
  #[clap(short, long)]
  image: PathBuf,
}

impl Predict {
  pub(crate) fn run(self) -> Result {
    let network = Network::load_weights(&self.weights.clone())?;

    let image = Self::read_image(self.image)?;

    let prediction = network.forward(image.view());

    println!(
      "Predicted digit: {}",
      argmax(&prediction.view().index_axis(Axis(1), 0))
    );

    Ok(())
  }

  fn read_image(path: PathBuf) -> Result<Array2<f64>> {
    let image_data = read(path)?;

    let image: image::ImageBuffer<image::Luma<u8>, Vec<u8>> =
      image::load_from_memory(&image_data)?.to_luma8();

    let (width, height) = image.dimensions();

    if width != 28 || height != 28 {
      bail!("image must be 28x28 pixels");
    }

    let flat_image: Vec<f64> = image
      .into_raw()
      .into_iter()
      .map(|p| p as f64 / 255.0)
      .collect();

    Ok(Array2::from_shape_vec((1, 784), flat_image)?)
  }
}
