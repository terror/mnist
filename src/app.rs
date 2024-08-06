use super::*;

pub struct App {
  network: Network,
  canvas: Array2<f64>,
  prediction: Option<usize>,
  is_drawing: bool,
}

impl eframe::App for App {
  fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
    egui::CentralPanel::default().show(ctx, |ui| {
      self.ui(ui);
    });
  }
}

impl App {
  pub fn new(network: Network) -> Self {
    Self {
      network,
      canvas: Array2::zeros((28, 28)),
      prediction: None,
      is_drawing: false,
    }
  }

  fn predict(&mut self) {
    let canvas = self.canvas.clone();

    let input = canvas.to_shape((1, 784)).unwrap();

    let output = self.network.forward(input.view());

    self.prediction =
      Some(crate::argmax(&output.view().index_axis(Axis(1), 0)));
  }

  fn clear_canvas(&mut self) {
    self.canvas.fill(0.0);
    self.prediction = None;
  }

  pub fn ui(&mut self, ui: &mut egui::Ui) {
    ui.heading("Draw a digit (0-9)");

    let (response, painter) =
      ui.allocate_painter(Vec2::new(280.0, 280.0), Sense::drag());

    let to_screen = emath::RectTransform::from_to(
      Rect::from_min_size(Pos2::ZERO, response.rect.size()),
      response.rect,
    );

    if self.is_drawing && response.dragged() {
      if let Some(pos) = response.interact_pointer_pos() {
        let canvas_pos = to_screen.inverse().transform_pos(pos);

        let (x, y) = (
          (canvas_pos.x / 10.0) as usize,
          (canvas_pos.y / 10.0) as usize,
        );

        if x < 28 && y < 28 {
          self.canvas[[y, x]] = 1.0;

          for dx in -1..=1 {
            for dy in -1..=1 {
              let nx = x as i32 + dx;

              let ny = y as i32 + dy;

              if nx >= 0 && nx < 28 && ny >= 0 && ny < 28 {
                self.canvas[[ny as usize, nx as usize]] =
                  self.canvas[[ny as usize, nx as usize]].max(0.5);
              }
            }
          }
        }
      }
    }

    if response.drag_started() {
      self.is_drawing = true;
    }

    if response.drag_stopped() {
      self.is_drawing = false;
      self.predict();
    }

    for y in 0..28 {
      for x in 0..28 {
        let color = Color32::from_gray((self.canvas[[y, x]] * 255.0) as u8);

        let min =
          to_screen.transform_pos(Pos2::new(x as f32 * 10.0, y as f32 * 10.0));

        let max = to_screen.transform_pos(Pos2::new(
          (x + 1) as f32 * 10.0,
          (y + 1) as f32 * 10.0,
        ));

        painter.rect_filled(Rect::from_two_pos(min, max), 0.0, color);
      }
    }

    ui.horizontal(|ui| {
      if ui.button("Clear").clicked() {
        self.clear_canvas();
      }

      if let Some(pred) = self.prediction {
        ui.label(format!("Prediction: {}", pred));
      }
    });
  }
}
