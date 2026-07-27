use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use eframe::egui::{
    ColorImage,
    Context as EguiContext,
    TextureHandle,
    TextureOptions,
};
use walkdir::WalkDir;

use crate::animation::{AnimationClip, AnimationConfig};

pub struct Assets {
    pub background: TextureHandle,
    pub screensavers: Vec<TextureHandle>,
    pub animations: HashMap<String, AnimationClip>,

    // Assets for the separate "Jine"-style chat window.
    pub chat_background: TextureHandle,
    pub chat_avatar: TextureHandle,

    // Assets for the "Task Manager" window (closeness/stress readout).
    pub closeness_icon: TextureHandle,
    pub stress_icon: TextureHandle,

    // Ame's stickers (LLM-picked) and pchan's stickers (manual picker),
    // keyed by filename stem, e.g. "aseru" -> aseru.png.
    pub ame_stickers: HashMap<String, TextureHandle>,
    pub pchan_stickers: HashMap<String, TextureHandle>,
}

impl Assets {
    pub fn load(ctx: &EguiContext) -> Result<Self> {
        let background = load_texture(
            ctx,
            "background",
            Path::new("assets/background/bg.png"),
        )?;

        let mut screensavers = Vec::new();

        for entry in WalkDir::new("assets/background/misc") {
            let entry = entry?;

            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) != Some("png") {
                continue;
            }

            let name = path
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();

            screensavers.push(load_texture(ctx, &name, path)?);
        }

        let animations =
            discover_animations(ctx, Path::new("assets/ame_sprite"))?;

        // NOTE: point these at your actual chat background / profile
        // picture files.
        let chat_background = load_texture(
            ctx,
            "chat_background",
            Path::new("assets/chat/background.png"),
        )?;

        let chat_avatar = load_texture(
            ctx,
            "chat_avatar",
            Path::new("assets/chat/avatar.png"),
        )?;

        // NOTE: point these at your actual Task Manager art (you said
        // you already have this).

        let closeness_icon = load_texture(
            ctx,
            "closeness_icon",
            Path::new("assets/task_manager/closeness_icon.png"),
        )?;

        let stress_icon = load_texture(
            ctx,
            "stress_icon",
            Path::new("assets/task_manager/stress_icon.png"),
        )?;

        // Sticker sets. Directory names match what you already have:
        // assets/chat/stickers/ame-only, assets/chat/stickers/pchanonly.
        let ame_stickers = load_stickers(
            ctx,
            Path::new("assets/chat/stickers/ame-only"),
        )?;

        let pchan_stickers = load_stickers(
            ctx,
            Path::new("assets/chat/stickers/pchanonly"),
        )?;

        Ok(Self {
            background,
            screensavers,
            animations,
            chat_background,
            chat_avatar,
            closeness_icon,
            stress_icon,
            ame_stickers,
            pchan_stickers,
        })
    }
}

fn load_stickers(ctx: &EguiContext, dir: &Path) -> Result<HashMap<String, TextureHandle>> {
    let mut stickers = HashMap::new();

    if !dir.exists() {
        // Not fatal — lets the app still run before sticker folders exist.
        eprintln!("Sticker folder missing, skipping: {:?}", dir);
        return Ok(stickers);
    }

    for entry in WalkDir::new(dir) {
        let entry = entry?;

        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) != Some("png") {
            continue;
        }

        let name = path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        stickers.insert(name.clone(), load_texture(ctx, &name, path)?);
    }

    Ok(stickers)
}

fn discover_animations(
    ctx: &EguiContext,
    root: &Path,
) -> Result<HashMap<String, AnimationClip>> {
    let mut folders: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();

    for entry in WalkDir::new(root) {
        let entry = entry?;

        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();

        if path.file_name().unwrap() == "animation.json" {
            continue;
        }

        if path.extension().and_then(|e| e.to_str()) != Some("png") {
            continue;
        }

        folders
            .entry(path.parent().unwrap().to_path_buf())
            .or_default()
            .push(path.to_path_buf());
    }

    let mut animations = HashMap::new();

    for (folder, mut files) in folders {
        // NOTE: this sorts by raw filename. If a folder has files with a
        // " #12345" suffix mixed in with clean "NNN.png" frames, this sort
        // order will be wrong (duplicate/orphaned exports sort ahead of the
        // real frame). Run tools/audit_frames.py to find offenders before
        // relying on frame order here.
        files.sort();

        let relative = folder
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', ".")
            .replace('/', ".");

        // "idle" -> category = name = "idle"
        // "idle_normal.normal_a" -> category = "idle_normal", name = "normal_a"
        let mut parts = relative.splitn(2, '.');
        let category = parts.next().unwrap_or(&relative).to_string();
        let name = parts.next().map(str::to_string).unwrap_or_else(|| category.clone());

        let config = load_animation_config(&folder)?;

        let mut frames = Vec::new();

        for (i, file) in files.iter().enumerate() {
            frames.push(load_texture(
                ctx,
                &format!("{}_{}", relative, i),
                file,
            )?);
        }

        let clip = AnimationClip {
            category,
            name,
            config,
            frames,
        };

        animations.insert(relative, clip);
    }

    Ok(animations)
}

fn load_animation_config(folder: &Path) -> Result<AnimationConfig> {
    let path = folder.join("animation.json");

    if !path.exists() {
        return Ok(AnimationConfig::default());
    }

    let text = fs::read_to_string(&path)
        .with_context(|| format!("Couldn't read {:?}", path))?;

    Ok(serde_json::from_str(&text)?)
}

fn load_texture(
    ctx: &EguiContext,
    name: &str,
    path: &Path,
) -> Result<TextureHandle> {
    let bytes = fs::read(path)
        .with_context(|| format!("Couldn't read {:?}", path))?;

    let image = image::load_from_memory(&bytes)?
        .to_rgba8();

    let size = [
        image.width() as usize,
        image.height() as usize,
    ];

    let pixels = image.into_vec();

    let image =
        ColorImage::from_rgba_unmultiplied(size, &pixels);

    Ok(ctx.load_texture(
        name,
        image,
        TextureOptions::NEAREST,
    ))
}