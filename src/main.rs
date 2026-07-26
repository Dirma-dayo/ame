mod animation;
mod app;
mod assets;
mod llm;
mod renderer;
mod state;

use app::PetApp;
use eframe::{
    egui,
    NativeOptions,
};

fn main() -> eframe::Result<()> {
    let options = NativeOptions {
    viewport: egui::ViewportBuilder::default()
        .with_title("Petto")
        .with_inner_size([696.0, 454.0])       // 2x native, just a nice default
        .with_min_inner_size([348.0, 227.0])   // don't allow shrinking below native
        .with_resizable(true)
        .with_drag_and_drop(false),
    ..Default::default()
};

    eframe::run_native(
        "Petto",
        options,
        Box::new(|cc| Ok(Box::new(PetApp::new(cc)))),
    )
}
