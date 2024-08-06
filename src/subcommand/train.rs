use super::*;

#[derive(Debug, Parser)]
pub(crate) struct Train {
  #[clap(short, long, default_value = "50")]
  epochs: usize,
  #[clap(short, long, default_value = "128")]
  batch_size: usize,
  #[clap(short, long, default_value = "weights.json")]
  output: PathBuf,
}

impl Train {
  pub(crate) fn run(self) -> Result {
    let mnist_data =
      Dataset::load("data").context("failed to load MNIST dataset")?;

    println!("Dataset loaded successfully:");
    println!("  Training images: {}", mnist_data.training_images.nrows());
    println!("  Training labels: {}", mnist_data.training_labels.nrows());
    println!("  Test images: {}", mnist_data.test_images.nrows());
    println!("  Test labels: {}", mnist_data.test_labels.nrows());

    let network = Arc::new(RwLock::new(Network::new(NetworkConfig::default())));

    let mut indices: Vec<usize> =
      (0..mnist_data.training_images.nrows()).collect();

    let progress_bar = ProgressBar::new(self.epochs as u64);

    progress_bar.set_style(
      ProgressStyle::default_bar()
        .template(
          "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} Epochs {msg}",
        )?
        .progress_chars("=>-"),
    );

    for _ in 0..self.epochs {
      indices.shuffle(&mut rand::thread_rng());

      indices.par_chunks(self.batch_size).for_each(|batch| {
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

    let accuracy = network
      .read()
      .unwrap()
      .evaluate(mnist_data.test_images.view(), mnist_data.test_labels.view());

    println!("Final accuracy: {:.2}%", accuracy * 100.0);

    network
      .read()
      .unwrap()
      .save_weights(&self.output)
      .context("failed to save network weights")?;

    println!("Saved weights to {}", self.output.display());

    Ok(())
  }
}
