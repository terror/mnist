use {super::*, app::App, predict::Predict, train::Train};

mod app;
mod predict;
mod train;

#[derive(Debug, Parser)]
pub(crate) enum Subcommand {
  App(App),
  Predict(Predict),
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
