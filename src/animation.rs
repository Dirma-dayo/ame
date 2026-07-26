use std::collections::HashMap;

use eframe::egui::TextureHandle;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct AnimationConfig {
    #[serde(default = "default_fps")]
    pub fps: f32,

    #[serde(default = "default_loop")]
    pub looping: bool,
}

fn default_fps() -> f32 {
    4.0
}

fn default_loop() -> bool {
    true
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            fps: default_fps(),
            looping: default_loop(),
        }
    }
}

pub struct AnimationClip {
    pub category: String,
    pub name: String,
    pub config: AnimationConfig,
    pub frames: Vec<TextureHandle>,
}

impl AnimationClip {
    pub fn id(&self) -> String {
        if self.category == self.name {
            self.name.clone()
        } else {
            format!("{}.{}", self.category, self.name)
        }
    }
}

pub struct AnimationPlayer {
    current: String,
    frame: usize,
    timer: f32,
}

impl AnimationPlayer {
    pub fn new(default: impl Into<String>) -> Self {
        Self {
            current: default.into(),
            frame: 0,
            timer: 0.0,
        }
    }

    pub fn play(&mut self, animation: impl Into<String>) {
        let animation = animation.into();

        if animation == self.current {
            return;
        }

        self.current = animation;
        self.frame = 0;
        self.timer = 0.0;
    }

    pub fn update(
        &mut self,
        delta: f32,
        animations: &HashMap<String, AnimationClip>,
    ) {
        let Some(animation) = animations.get(&self.current) else {
            return;
        };

        if animation.frames.len() <= 1 {
            return;
        }

        self.timer += delta;
        let frame_time = 1.0 / animation.config.fps;

        while self.timer >= frame_time {
            self.timer -= frame_time;
            self.frame += 1;

            if self.frame >= animation.frames.len() {
                if animation.config.looping {
                    self.frame = 0;
                } else {
                    self.frame = animation.frames.len() - 1;
                }
            }
        }
    }

    pub fn texture<'a>(
        &self,
        animations: &'a HashMap<String, AnimationClip>,
    ) -> Option<&'a TextureHandle> {
        animations
            .get(&self.current)
            .and_then(|animation| animation.frames.get(self.frame))
    }

    pub fn current(&self) -> &str {
        &self.current
    }

    pub fn frame(&self) -> usize {
        self.frame
    }
}