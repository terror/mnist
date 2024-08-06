use {super::*, app::App, predict::Predict, train::Train};

mod app;
mod predict;
mod train;

#[derive(Debug, Parser)]
pub(crate) enum Subcommand {
  #[clap(name = "app", about = "Run an interactive GUI application")]
  App(App),
  #[clap(name = "predict", about = "Predict the class of a new instance")]
  Predict(Predict),
  #[clap(name = "train", about = "Train the model")]
  Train(Train),
}

impl Subcommand {
  pub(crate) fn run(self) -> Result {
    match self {
      Self::App(app) => app.run(),
      Self::Predict(predict) => predict.run(),
      Self::Train(train) => train.run(),
    }
  }
}
