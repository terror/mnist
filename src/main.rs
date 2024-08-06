use {
  crate::{
    arguments::Arguments, config::*, dataset::Dataset, math::*,
    network::Network, subcommand::Subcommand,
  },
  anyhow::{bail, Context},
  clap::Parser,
  indicatif::{ProgressBar, ProgressStyle},
  ndarray::{Array2, ArrayView, ArrayView2, Axis},
  ndarray_rand::{rand_distr::Uniform, RandomExt},
  rand::seq::SliceRandom,
  rayon::prelude::*,
  serde::{Deserialize, Serialize},
  std::{
    fs::{read, File},
    io::Read,
    path::Path,
    path::PathBuf,
    process,
    sync::{Arc, RwLock},
  },
};

mod arguments;
mod config;
mod dataset;
mod math;
mod network;
mod subcommand;

type Result<T = (), E = anyhow::Error> = std::result::Result<T, E>;

fn main() {
  if let Err(error) = Arguments::parse().run() {
    eprintln!("error: {error}");
    process::exit(1);
  }
}
