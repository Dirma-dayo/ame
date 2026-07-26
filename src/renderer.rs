use crate::{animation::AnimationPlayer, assets::Assets};
use eframe::egui::{
    self, Color32, Context, Frame, Image, Pos2, Rect, Sense, TextureOptions, Vec2,
};

pub struct Renderer;

impl Renderer {
    /// True native resolution of the art. Everything else is derived from this.
    const ART_SIZE: Vec2 = Vec2::new(348.0, 227.0);

    pub fn draw(ctx: &Context, assets: &Assets, player: &AnimationPlayer, reply: Option<&str>) {
        // Crisp edges on rounded rects / lines (bubble frame etc.)
        ctx.tessellation_options_mut(|to| to.feathering = false);

        let screen = ctx.screen_rect();

        // Largest uniform scale that fits the window without cropping.
        let scale = (screen.width() / Self::ART_SIZE.x)
            .min(screen.height() / Self::ART_SIZE.y)
            .max(0.01);

        let display_size = Self::ART_SIZE * scale;
        let origin = Pos2::new(
            (screen.width() - display_size.x) * 0.5,
            (screen.height() - display_size.y) * 0.5,
        );
        let canvas_rect = Rect::from_min_size(origin, display_size);

        egui::Area::new("pet".into())
            .fixed_pos(Pos2::ZERO)
            .show(ctx, |ui| {
                // Letterbox bars for whatever isn't covered by canvas_rect.
                ui.painter().rect_filled(screen, 0.0, Color32::BLACK);

                // Background
                ui.put(
                    canvas_rect,
                    Image::new(&assets.background)
                        .texture_options(TextureOptions::NEAREST)
                        .fit_to_exact_size(display_size),
                );

                // Current animation frame
                if let Some(texture) = player.texture(&assets.animations) {
                    // Sprite was authored at half the old canvas's width/height;
                    // keep that same proportion relative to native art size.
                    let sprite_native = Self::ART_SIZE * 1.0; // 2x native, just a nice default
                    let sprite_size = sprite_native * scale;
                    let pos = Pos2::new(
                        origin.x + (display_size.x - sprite_size.x) * 0.5,
                        origin.y + display_size.y - sprite_size.y,
                    );

                    let response = ui.put(
                        Rect::from_min_size(pos, sprite_size),
                        Image::new(texture)
                            .texture_options(TextureOptions::NEAREST)
                            .fit_to_exact_size(sprite_size)
                            .sense(Sense::click_and_drag()),
                    );

                    if response.dragged() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                    }
                }

                // Speech bubble for the latest LLM reply
                if let Some(text) = reply {
                    let bubble_margin = 12.0 * scale;
                    let bubble_rect = Rect::from_min_size(
                        origin + Vec2::new(24.0, 16.0) * scale,
                        Vec2::new(648.0, 70.0) * scale,
                    );
                    let text_width = bubble_rect.width() - bubble_margin * 2.0;

                    ui.put(bubble_rect, |ui: &mut egui::Ui| {
                        Frame::default()
                            .fill(Color32::from_black_alpha(190))
                            .corner_radius(10.0 * scale)
                            .inner_margin(bubble_margin)
                            .show(ui, |ui| {
                                // A plain `colored_label` here defaults to
                                // "extend" wrap behavior inside this custom
                                // `ui.put` closure, so long replies just ran
                                // past the bubble's edges instead of
                                // breaking onto new lines. Giving the label
                                // its own width-bounded child `Ui` plus an
                                // explicit wrap mode fixes that.
                                ui.allocate_ui_with_layout(
                                    Vec2::new(text_width, 0.0),
                                    egui::Layout::top_down(egui::Align::LEFT),
                                    |ui| {
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(text).color(Color32::WHITE),
                                            )
                                            .wrap_mode(egui::TextWrapMode::Wrap),
                                        );
                                    },
                                );
                            })
                            .response
                    });
                }

                // Future random event overlays go here, using `origin`/`scale`
                // the same way as the sprite above.
            });

        ctx.request_repaint();
    }
}