use super::*;

#[derive(Debug, Parser)]
pub(crate) struct App {
  #[clap(short, long)]
  weights: PathBuf,
}

impl App {
  pub(crate) fn run(self) -> Result {
    let network = Network::load_weights(&self.weights)?;

    let app = Interface::new(network);

    let native_options = NativeOptions {
      centered: true,
      hardware_acceleration: HardwareAcceleration::Preferred,
      viewport: ViewportBuilder {
        max_inner_size: Some(egui::vec2(450.0, 350.0)),
        ..Default::default()
      },
      ..Default::default()
    };

    eframe::run_native(
      env!("CARGO_PKG_NAME"),
      native_options,
      Box::new(|_| Ok(Box::new(app))),
    )
    .unwrap();

    Ok(())
  }
}
