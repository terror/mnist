use {super::*, predict::Predict, train::Train};

mod predict;
mod train;

#[derive(Debug, Parser)]
pub(crate) enum Subcommand {
  Predict(Predict),
  Train(Train),
}

impl Subcommand {
  pub(crate) fn run(self) -> Result {
    match self {
      Self::Predict(predict) => predict.run(),
      Self::Train(train) => train.run(),
    }
  }
}
