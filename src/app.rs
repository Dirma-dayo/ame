use std::collections::VecDeque;
use chrono::Timelike;

use crate::{
    animation::AnimationPlayer,
    assets::Assets,
    llm::{
        list_models, load_example_dialogue, parse_llm_reply, ChatMessage, LlmWorker,
        DEFAULT_BASE_URL,
    },
    renderer::Renderer,
    state::{Activity, Mood, PetState},
};
use eframe::egui::{self, Context};

// Hard cap on how many entries the raw debug log keeps around. Log
// entries only contain the *new* messages since the last send (see
// `log_sent`), so this is mostly a safety net against pathological
// growth rather than the primary size control.
const MAX_RAW_LOG_ENTRIES: usize = 200;

// Sticker sets. Ame's are picked autonomously by the LLM (see the
// system prompt / `LlmReply::sticker`); pchan's are sent manually via
// the picker in the chat window's input bar. These lists exist so the
// system prompt can enumerate exactly what Ame is allowed to pick, and
// so the picker UI has a fixed, ordered set to render — the actual
// texture lookup still goes through `Assets::ame_stickers` /
// `Assets::pchan_stickers`, so a name here with no matching PNG just
// quietly fails to render rather than panicking.
const AME_STICKERS: &[&str] = &[
    "aseru", "ignoring_u", "im_orb", "selfie_time", "so_NOT_cute", "toketeiru", "zzz",
];
const PCHAN_STICKERS: &[&str] = &[
    "idc", "im_ded", "love_forever", "omg", "sad", "sorry", "this", "tired_ok",
];

// Sticker images are rendered at a fixed size in the chat log,
// independent of the text bubble width.
const STICKER_SIZE: f32 = 96.0;

/// UI-facing chat entry. Separate from `ChatMessage` (the LLM wire
/// format) because a bubble can now be plain text or a sticker image,
/// and only text ever needs to round-trip through the model as prose.
#[derive(Debug, Clone)]
enum DisplayContent {
    Text(String),
    Sticker(String), // file stem, e.g. "aseru"
}

#[derive(Debug, Clone)]
struct DisplayMessage {
    role: &'static str, // "user" | "assistant"
    content: DisplayContent,
}

impl DisplayMessage {
    fn user_text(s: impl Into<String>) -> Self {
        Self {
            role: "user",
            content: DisplayContent::Text(s.into()),
        }
    }
    fn assistant_text(s: impl Into<String>) -> Self {
        Self {
            role: "assistant",
            content: DisplayContent::Text(s.into()),
        }
    }
    fn user_sticker(name: impl Into<String>) -> Self {
        Self {
            role: "user",
            content: DisplayContent::Sticker(name.into()),
        }
    }
    fn assistant_sticker(name: impl Into<String>) -> Self {
        Self {
            role: "assistant",
            content: DisplayContent::Sticker(name.into()),
        }
    }
}

pub struct PetApp {
    assets: Assets,
    state: PetState,
    player: AnimationPlayer,

    llm: LlmWorker,
    available_models: Vec<String>,
    selected_model: String,

    // If empty, `effective_base_url()` falls back to `DEFAULT_BASE_URL`.
    // `proxy_input` is a separate scratch buffer so typing doesn't
    // rebuild the worker/model list on every keystroke — only on Apply.
    proxy: String,
    proxy_input: String,
    settings_open: bool,

    // LLM-facing history: [system prompt, ...real turns]. Assistant
    // turns here are stored as the exact JSON envelope the model was
    // asked to produce (see `poll_llm`), as ONE turn per LLM response —
    // never split into one turn per display bubble. Keeping this
    // consistent with what the system prompt + few-shot examples teach
    // stops the model's own past replies from reinforcing drift away
    // from the required format.
    conversation: Vec<ChatMessage>,
    examples: Vec<ChatMessage>,

    // UI-facing history for the Jine chat window: one entry per bubble,
    // either plain text or a sticker image (no JSON envelope, no
    // system-role entries). Deliberately a separate list from
    // `conversation`.
    display_log: Vec<DisplayMessage>,

    input: String,
    waiting_for_reply: bool,

    // Whether the separate "Jine"-style chat window is currently shown.
    chat_window_open: bool,
    // Whether the "Task Manager" (closeness/stress) window is shown.
    task_manager_open: bool,
    // Whether pchan's sticker picker popup is currently shown.
    sticker_picker_open: bool,

    // Hidden debug window: raw JSON sent to / received from the LLM.
    // Not shown by default — toggled with 'O'.
    raw_history_open: bool,
    // Logged exchanges, oldest first. Each SENT entry only contains the
    // messages appended since the previous send (not the whole growing
    // history) to avoid the log — and the redraw cost of showing it —
    // growing quadratically with conversation length.
    raw_log: Vec<String>,
    // How many messages of `conversation` had already been logged as of
    // the last `log_sent` call. Drives the "new messages only" delta.
    logged_message_count: usize,

    // Extra bubbles queued up after the first bubble of a multi-bubble
    // ("double/triple texting") reply has already been shown.
    pending_bubbles: VecDeque<String>,
    bubble_timer: f32,
}

impl PetApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let assets = Assets::load(&cc.egui_ctx).expect("Couldn't load assets");
        let state = PetState::new();
        let player = AnimationPlayer::new(state.current_animation());

        let conversation = vec![ChatMessage::system(
            "You are Ame, a small desktop companion living on someone's \
             screen. Reply in 1-2 short, casual sentences most of the \
             time. Warm, a little playful, never robotic. \
             Occasionally (roughly 1 in every 5-6 replies, never forced) \
             if it genuinely fits the moment — excitement, rambling, a \
             joke with a setup and punchline — you may instead send a \
             quick back-to-back burst of 2 to 10 very short messages, \
             like someone texting in a row, by making \"reply\" a JSON \
             array of strings instead of a single string. Most replies \
             should still be a single string. \
             do NOT ever send a JSON array of objects, only a single object with a \"reply\" \
             do NOT ever use emoji \
             You also have a small set of stickers you can send: aseru, \
             ignoring_u, im_orb, selfie_time, so_NOT_cute, toketeiru, zzz \
             — think of them as short mood/reaction images. Include a \
             \"sticker\" field ONLY when one genuinely fits the moment; \
             this should be rare, most replies send no sticker at all. \
             Omit the field or set it to null otherwise. Never invent a \
             sticker name outside this exact list. If a message tells \
             you the user sent a sticker, react to it the way you'd \
             react to a little picture they shared. \
             Before most replies you'll get a system note with the real \
             current time and day. Only bring it up when it actually fits \
             — being sleepy late at night, greeting a new day, noting \
             it's the weekend — most replies shouldn't mention it at all, \
             and you should never state the exact time/date like a clock \
             or acknowledge that you were told it. \
             Sometimes you'll get a system note saying the user hasn't \
             talked to you in a while — when that happens, initiate a \
             short, natural message first, as if reaching out on your \
             own. Never mention the note itself. \
             You MUST respond with ONLY a JSON object, no other text, no \
             markdown fences, in exactly this shape: \
             {\"reply\": \"<your 1-2 sentence reply>\" or [\"<msg1>\", \
             \"<msg2>\", ...], \"mood\": \"<one of: worried, excited, \
             disappointed, pissed_off, neutral>\", \"sticker\": \"<one \
             of the sticker names above>\" or null}",
        )];

        let examples = load_example_dialogue("assets/examples.txt")
            .unwrap_or_else(|err| {
                eprintln!("Couldn't load examples.txt: {err}");
                Vec::new()
            });

        // No proxy configured yet at startup, so we fall back to the
        // default local endpoint.
        let proxy = String::new();
        let base_url = effective_base_url(&proxy);

        let available_models = list_models(&base_url).unwrap_or_else(|_| {
            vec!["llama3.2".to_string()]
        });

        let selected_model = available_models[0].clone();

        Self {
            assets,
            state,
            player,

            llm: LlmWorker::new(selected_model.clone(), base_url),

            available_models,
            selected_model,
            proxy: proxy.clone(),
            proxy_input: proxy,
            settings_open: false,
            conversation,
            examples,
            display_log: Vec::new(),
            input: String::new(),

            waiting_for_reply: false,
            chat_window_open: true,
            task_manager_open: true,
            sticker_picker_open: false,

            raw_history_open: false,
            raw_log: Vec::new(),
            logged_message_count: 0,

            pending_bubbles: VecDeque::new(),
            bubble_timer: 0.0,
        }
    }

    /// Re-points the worker at the new proxy (or back to the default if
    /// cleared), refetches the model list against it, and rebuilds the
    /// worker thread. Only called from the settings window's Apply button.
    fn apply_proxy(&mut self) {
        self.proxy = self.proxy_input.trim().to_string();
        let base_url = effective_base_url(&self.proxy);

        self.available_models = list_models(&base_url).unwrap_or_else(|_| {
            vec![self.selected_model.clone()]
        });

        if !self.available_models.contains(&self.selected_model) {
            if let Some(first) = self.available_models.first() {
                self.selected_model = first.clone();
            }
        }

        self.llm = LlmWorker::new(self.selected_model.clone(), base_url);
    }

    fn change_model(&mut self, model: String) {
        self.selected_model = model.clone();
        self.llm = LlmWorker::new(model, effective_base_url(&self.proxy));
    }

    /// Logs only the messages appended to `conversation` since the last
    /// call — not the whole array — so the debug log (and its per-frame
    /// redraw cost) grows roughly linearly with turn count instead of
    /// quadratically. The very first call logs the full initial context
    /// (system prompt + examples + first user message) since nothing's
    /// been logged yet.
    fn log_sent(&mut self, messages: &[ChatMessage]) {
        let n = self.raw_log.len();
        let total = messages.len();
        let start = self.logged_message_count.min(total);
        let new_messages = &messages[start..];

        let json = serde_json::to_string_pretty(new_messages)
            .unwrap_or_else(|err| format!("<failed to serialize: {err}>"));

        let label = if start == 0 {
            format!(">>> [{n}] SENT — full context ({total} messages)")
        } else {
            format!(
                ">>> [{n}] SENT — {} new message(s) appended (total {total}, {start} unchanged since last send)",
                new_messages.len()
            )
        };

        self.raw_log.push(format!("{label}\n{json}"));
        self.logged_message_count = total;
        self.trim_raw_log();
    }

    /// Logs the raw, pre-parse response text (or error) exactly as it
    /// came back from Ollama, prefixed with a `<<<` marker.
    fn log_received(&mut self, raw: &str) {
        let n = self.raw_log.len();
        self.raw_log.push(format!("<<< [{n}] RECEIVED (raw)\n{raw}"));
        self.trim_raw_log();
    }

    fn log_error(&mut self, err: &str) {
        let n = self.raw_log.len();
        self.raw_log.push(format!("<<< [{n}] ERROR\n{err}"));
        self.trim_raw_log();
    }

    /// Safety-net cap: even with delta-only logging, drop the oldest
    /// entries once the log gets unreasonably long.
    fn trim_raw_log(&mut self) {
        if self.raw_log.len() > MAX_RAW_LOG_ENTRIES {
            let excess = self.raw_log.len() - MAX_RAW_LOG_ENTRIES;
            self.raw_log.drain(0..excess);
        }
    }

    fn send_message(&mut self) {
        let text = self.input.trim().to_string();

        if text.is_empty() || self.waiting_for_reply {
            return;
        }

        self.input.clear();
        self.conversation.push(ChatMessage::user(text.clone()));
        self.display_log.push(DisplayMessage::user_text(text));
        self.state.record_chat_interaction();

        // system prompt + few-shot examples + real conversation history.
        // self.conversation stays as just [system, ...real history] so
        // examples never get saved/duplicated across turns.
        let mut messages = vec![self.conversation[0].clone()];
        messages.extend(self.examples.clone());
        messages.extend(self.conversation[1..].to_vec());
        messages.push(ChatMessage::system(current_time_context()));

        self.log_sent(&messages);
        self.llm.ask(messages);

        self.waiting_for_reply = true;
        self.state.set_activity(Activity::PhoneChat);
    }

    /// pchan picked a sticker from the picker. Tells the model a sticker
    /// was sent (as a plain text note, since the LLM only deals in
    /// prose/JSON) and renders the actual image in `display_log`.
    fn send_sticker(&mut self, name: String) {
        if self.waiting_for_reply {
            return;
        }

        self.conversation
            .push(ChatMessage::user(format!("[pchan sent a sticker: {name}]")));
        self.display_log.push(DisplayMessage::user_sticker(name));
        self.state.record_chat_interaction();

        let mut messages = vec![self.conversation[0].clone()];
        messages.extend(self.examples.clone());
        messages.extend(self.conversation[1..].to_vec());

        self.log_sent(&messages);
        self.llm.ask(messages);

        self.waiting_for_reply = true;
        self.state.set_activity(Activity::PhoneChat);
    }

    /// Ame speaking up first because the user's gone quiet for a while.
    /// Injects a system-role note into `conversation` only (it never
    /// renders as a fake bubble since `display_log` is separate) and
    /// otherwise reuses the normal ask path.
    fn initiate_chat(&mut self, idle_minutes: f32) {
        if self.waiting_for_reply {
            return;
        }

        let nudge = format!(
            "[System note: the user hasn't talked to you in about {:.0} \
             minute(s). Initiate a short, casual message to them first — \
             check in, share a passing thought, or ask something. Don't \
             mention this note.]",
            idle_minutes
        );
        self.conversation.push(ChatMessage::system(nudge));

        let mut messages = vec![self.conversation[0].clone()];
        messages.extend(self.examples.clone());
        messages.extend(self.conversation[1..].to_vec());

        self.log_sent(&messages);
        self.llm.ask(messages);

        self.waiting_for_reply = true;
        self.state.set_activity(Activity::PhoneChat);
    }

    fn poll_llm(&mut self) {
        let Some(result) = self.llm.poll() else {
            return;
        };

        self.waiting_for_reply = false;

        match result {
            Ok(raw) => {
                self.log_received(&raw);

                // Tolerant parse: handles a proper {"reply":...,"mood":...}
                // object, a bare ["msg1","msg2"] array, a bare string, or
                // trailing garbage after any of those — see
                // `parse_llm_reply` for why that's needed.
                let parsed = parse_llm_reply(&raw);
                let mood = Mood::from_llm(&parsed.mood);

                // Store the reply back into LLM-facing history in the
                // exact JSON envelope we asked for, as ONE turn — not
                // one turn per display bubble. This keeps history
                // consistent with the system prompt and few-shot
                // examples, instead of teaching the model that several
                // consecutive assistant turns with no user message in
                // between are a normal thing to do.
                let canonical =
                    serde_json::to_string(&parsed).unwrap_or_else(|_| raw.clone());
                self.conversation.push(ChatMessage::assistant(canonical));

                let mut bubbles = parsed.reply.into_bubbles();
                if bubbles.is_empty() {
                    bubbles.push(raw.clone());
                }

                let first = bubbles.remove(0);

                self.display_log
                    .push(DisplayMessage::assistant_text(first.clone()));
                self.state.show_reply(first, 6.0);
                self.state.set_mood(mood, &self.assets.animations);

                // Only render it if it's a name we actually have an
                // asset for — a small local model can still hallucinate
                // outside the allowed list.
                if let Some(sticker) = parsed.sticker.as_deref() {
                    if self.assets.ame_stickers.contains_key(sticker) {
                        self.display_log
                            .push(DisplayMessage::assistant_sticker(sticker.to_string()));
                    }
                }

                self.pending_bubbles = bubbles.into_iter().collect();
                self.bubble_timer = 0.0; // fire first queued bubble soon
            }
            Err(err) => {
                self.log_error(&err.to_string());

                // Keep it in-character rather than showing a raw error.
                self.state.show_reply(
                    format!("...hmm, something's wrong ({err})"),
                    4.0,
                );
                self.state.set_mood(Mood::Neutral, &self.assets.animations);
            }
        }
    }

    /// Pops the next queued bubble (from a multi-bubble reply) onto
    /// screen once its little "typing" delay has elapsed. Only touches
    /// `display_log` — the LLM-facing `conversation` already got the
    /// full canonical reply as a single turn back in `poll_llm`.
    fn advance_pending_bubbles(&mut self, delta: f32) {
        if self.pending_bubbles.is_empty() {
            return;
        }

        self.bubble_timer -= delta;
        if self.bubble_timer > 0.0 {
            return;
        }

        if let Some(next) = self.pending_bubbles.pop_front() {
            self.bubble_timer = typing_delay(&next);
            self.display_log
                .push(DisplayMessage::assistant_text(next.clone()));
            self.state.show_reply(next, 6.0);
        }
    }

    /// Opens (or keeps open) the "Jine"-style chat window as a separate
    /// OS-level window alongside the pet window. Re-shown every frame,
    /// same pattern egui uses for immediate-mode multi-viewport apps.
    fn draw_chat_window(&mut self, ctx: &Context) {
        if !self.chat_window_open {
            return;
        }

        let viewport_id = egui::ViewportId::from_hash_of("chat_window");

        let builder = egui::ViewportBuilder::default()
            .with_title("Jine")
            .with_inner_size([360.0, 480.0])
            .with_min_inner_size([300.0, 360.0])
            .with_decorations(false) // we paint our own titlebar below
            .with_resizable(true)
            .with_transparent(false);

        let mut should_close = false;

        ctx.show_viewport_immediate(viewport_id, builder, |ctx, _class| {
            // Without this, only the root (pet) window keeps re-arming
            // its own repaint each frame. This viewport's repaint isn't
            // flagged on its own, so after losing OS focus egui stops
            // reliably repainting/processing input for it — clicks and
            // typing appear "dead" until the viewport is torn down and
            // rebuilt. Requesting repaint from inside this closure flags
            // *this* viewport specifically, keeping it live regardless
            // of focus.
            ctx.request_repaint();

            if ctx.input(|i| i.viewport().close_requested()) {
                should_close = true;
            }

            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    self.chat_ui(ui, ctx, &mut should_close);
                });
        });

        if should_close {
            self.chat_window_open = false;
        }
    }

    /// The actual contents of the chat window: hand-painted titlebar
    /// (with the settings gear now docked into it), scrolling message
    /// log with avatar/bubbles/stickers, a sticker picker popup, and a
    /// bottom bar with the model picker + sticker button + text input.
    fn chat_ui(&mut self, ui: &mut egui::Ui, ctx: &Context, should_close: &mut bool) {
        let rect = ui.max_rect();

        // Background art, stretched to fill the window.
        egui::Image::new(&self.assets.chat_background)
            .texture_options(egui::TextureOptions::NEAREST)
            .paint_at(ui, rect);

        let title_bar_rect =
            draw_mini_titlebar(ui, ctx, rect, "chat", "JINE", should_close);

        // Settings gear, docked into the titlebar just left of the
        // minimize/maximize/close cluster (moved here from the input
        // bar so it reads as window chrome rather than a chat control).
        {
            let gear_size = egui::vec2(20.0, 18.0);
            let controls_width = (20.0 + 4.0) * 3.0; // matches the 3 buttons drawn in draw_mini_titlebar
            let gear_x = title_bar_rect.max.x - 6.0 - controls_width - 6.0 - gear_size.x;
            let gear_rect = egui::Rect::from_min_size(
                egui::pos2(gear_x, title_bar_rect.min.y + 5.0),
                gear_size,
            );
            let resp = ui.interact(
                gear_rect,
                ui.id().with(("chat", "titlebar_settings")),
                egui::Sense::click(),
            );
            let base = egui::Color32::from_rgb(90, 90, 130);
            let color = if resp.hovered() {
                base.gamma_multiply(1.3)
            } else {
                base
            };
            ui.painter().rect_filled(gear_rect, 3.0, color);
            ui.painter().text(
                gear_rect.center(),
                egui::Align2::CENTER_CENTER,
                "\u{2699}", // ⚙
                egui::FontId::monospace(12.0),
                egui::Color32::WHITE,
            );
            if resp.clicked() {
                self.proxy_input = self.proxy.clone();
                self.settings_open = true;
            }
        }

        // --- Body: chat log + input bar ---------------------------------
        let body_rect =
            egui::Rect::from_min_max(egui::pos2(rect.min.x, title_bar_rect.max.y), rect.max);

        let input_bar_height = 64.0;
        let chat_area_rect = egui::Rect::from_min_max(
            body_rect.min,
            egui::pos2(body_rect.max.x, body_rect.max.y - input_bar_height),
        );
        let input_rect = egui::Rect::from_min_max(
            egui::pos2(body_rect.min.x, body_rect.max.y - input_bar_height),
            body_rect.max,
        );

        let mut chat_ui = ui.new_child(egui::UiBuilder::new().max_rect(chat_area_rect));
        egui::ScrollArea::vertical()
            .id_salt("chat_scroll")
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(&mut chat_ui, |ui| {
                ui.add_space(12.0);
                ui.set_width(chat_area_rect.width());

                // display_log holds only real user/assistant turns — no
                // system prompt, no JSON envelope, no system-role
                // idle-initiation nudges — so no filtering needed here.
                for message in self.display_log.iter() {
                    match (&message.content, message.role) {
                        (DisplayContent::Text(text), "user") => draw_user_bubble(ui, text),
                        (DisplayContent::Text(text), "assistant") => {
                            draw_assistant_bubble(ui, &self.assets.chat_avatar, text)
                        }
                        (DisplayContent::Sticker(name), "user") => {
                            if let Some(tex) = self.assets.pchan_stickers.get(name.as_str()) {
                                draw_user_sticker(ui, tex);
                            }
                        }
                        (DisplayContent::Sticker(name), "assistant") => {
                            if let Some(tex) = self.assets.ame_stickers.get(name.as_str()) {
                                draw_assistant_sticker(ui, &self.assets.chat_avatar, tex);
                            }
                        }
                        _ => {}
                    }
                    ui.add_space(8.0);
                }

                if self.waiting_for_reply {
                    draw_assistant_bubble(ui, &self.assets.chat_avatar, "...");
                }
            });

        let mut input_ui = ui.new_child(egui::UiBuilder::new().max_rect(input_rect));
        egui::Frame::default()
            .fill(egui::Color32::from_rgba_unmultiplied(20, 20, 40, 215))
            .inner_margin(8.0)
            .show(&mut input_ui, |ui| {
                let previous = self.selected_model.clone();

                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("model_select")
                        .selected_text(&self.selected_model)
                        .show_ui(ui, |ui| {
                            for model in &self.available_models {
                                ui.selectable_value(
                                    &mut self.selected_model,
                                    model.clone(),
                                    model,
                                );
                            }
                        });

                    if ui
                        .button("\u{1F380}") // 🎀
                        .on_hover_text("Send a sticker")
                        .clicked()
                    {
                        self.sticker_picker_open = !self.sticker_picker_open;
                    }
                });

                if previous != self.selected_model {
                    self.change_model(self.selected_model.clone());
                }

                ui.horizontal(|ui| {
                    let response = ui.add_sized(
                        [ui.available_width() - 60.0, 24.0],
                        egui::TextEdit::singleline(&mut self.input)
                            .hint_text("Say something..."),
                    );

                    let enter_pressed =
                        response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

                    let send_clicked = ui
                        .add_enabled(!self.waiting_for_reply, egui::Button::new("Send"))
                        .clicked();

                    if send_clicked || enter_pressed {
                        self.send_message();
                    }
                });
            });

        // pchan's manual sticker picker, anchored just above the input
        // bar. Separate from the input frame above so it floats over
        // the chat log instead of squeezing the layout.
        if self.sticker_picker_open {
            let mut close_picker = false;
            let mut clicked: Option<String> = None;

            egui::Window::new("pchan_sticker_picker")
                .title_bar(false)
                .resizable(false)
                .collapsible(false)
                .fixed_size([200.0, 190.0])
                .anchor(
                    egui::Align2::RIGHT_BOTTOM,
                    egui::vec2(-8.0, -(input_bar_height + 8.0)),
                )
                .show(ctx, |ui| {
                    egui::Frame::default()
                        .fill(egui::Color32::from_rgba_unmultiplied(24, 24, 44, 240))
                        .corner_radius(8.0)
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new("Send a sticker")
                                        .color(egui::Color32::WHITE)
                                        .size(12.0),
                                );
                                if ui.small_button("\u{2715}").clicked() {
                                    close_picker = true;
                                }
                            });
                            ui.separator();
                            ui.horizontal_wrapped(|ui| {
                                for name in PCHAN_STICKERS {
                                    if let Some(tex) = self.assets.pchan_stickers.get(*name) {
                                        let resp =
                                            ui.add(egui::ImageButton::new(tex).frame(false));
                                        if resp.clicked() {
                                            clicked = Some((*name).to_string());
                                        }
                                    }
                                }
                            });
                        });
                });

            if close_picker {
                self.sticker_picker_open = false;
            }
            if let Some(name) = clicked {
                self.sticker_picker_open = false;
                self.send_sticker(name);
            }
        }
    }

    /// Opens (or keeps open) the "Task Manager" window: a small always-
    /// on readout of closeness/stress, styled after the reference
    /// screenshot. Same separate-OS-window pattern as the chat window.
    fn draw_task_manager_window(&mut self, ctx: &Context) {
        if !self.task_manager_open {
            return;
        }

        let viewport_id = egui::ViewportId::from_hash_of("task_manager_window");

        let builder = egui::ViewportBuilder::default()
            .with_title("Task Manager")
            .with_inner_size([320.0, 240.0])
            .with_min_inner_size([280.0, 200.0])
            .with_decorations(false)
            .with_resizable(true)
            .with_transparent(false);

        let mut should_close = false;

        ctx.show_viewport_immediate(viewport_id, builder, |ctx, _class| {
            ctx.request_repaint();

            if ctx.input(|i| i.viewport().close_requested()) {
                should_close = true;
            }

            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    self.task_manager_ui(ui, ctx, &mut should_close);
                });
        });

        if should_close {
            self.task_manager_open = false;
        }
    }

    /// Small settings popup: just the proxy field for now. Same
    /// chrome-less-mini-window pattern as chat/task-manager.
    fn draw_settings_window(&mut self, ctx: &Context) {
        if !self.settings_open {
            return;
        }

        let viewport_id = egui::ViewportId::from_hash_of("settings_window");

        let builder = egui::ViewportBuilder::default()
            .with_title("Settings")
            .with_inner_size([340.0, 160.0])
            .with_min_inner_size([300.0, 140.0])
            .with_decorations(false)
            .with_resizable(true)
            .with_transparent(false);

        let mut should_close = false;

        ctx.show_viewport_immediate(viewport_id, builder, |ctx, _class| {
            ctx.request_repaint();

            if ctx.input(|i| i.viewport().close_requested()) {
                should_close = true;
            }

            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    let rect = ui.max_rect();
                    ui.painter()
                        .rect_filled(rect, 0.0, egui::Color32::from_rgb(24, 28, 48));

                    let title_bar_rect =
                        draw_mini_titlebar(ui, ctx, rect, "settings", "SETTINGS", &mut should_close);

                    let body_rect = egui::Rect::from_min_max(
                        egui::pos2(rect.min.x, title_bar_rect.max.y),
                        rect.max,
                    );
                    let mut body_ui = ui.new_child(egui::UiBuilder::new().max_rect(body_rect));
                    body_ui.add_space(16.0);
                    body_ui.horizontal(|ui| ui.add_space(12.0));

                    egui::Frame::default()
                        .inner_margin(12.0)
                        .show(&mut body_ui, |ui| {
                            ui.label(
                                egui::RichText::new("Proxy")
                                    .color(egui::Color32::WHITE)
                                    .strong(),
                            );
                            ui.add_space(4.0);
                            ui.add_sized(
                                [ui.available_width(), 22.0],
                                egui::TextEdit::singleline(&mut self.proxy_input)
                                    .hint_text(DEFAULT_BASE_URL),
                            );
                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new(format!(
                                    "Leave empty to use the default ({DEFAULT_BASE_URL})"
                                ))
                                .weak()
                                .size(11.0),
                            );
                            ui.add_space(10.0);

                            ui.horizontal(|ui| {
                                if ui.button("Apply").clicked() {
                                    self.apply_proxy();
                                }
                                if ui.button("Close").clicked() {
                                    should_close = true;
                                }
                            });
                        });
                });
        });

        if should_close {
            self.settings_open = false;
        }
    }

    fn task_manager_ui(&mut self, ui: &mut egui::Ui, ctx: &Context, should_close: &mut bool) {
        let rect = ui.max_rect();

        ui.painter().rect_filled(rect, 0.0, egui::Color32::WHITE);

        let title_bar_rect =
            draw_mini_titlebar(ui, ctx, rect, "task_manager", "TASK MANAGER", should_close);

        let body_rect =
            egui::Rect::from_min_max(egui::pos2(rect.min.x, title_bar_rect.max.y), rect.max);

        let mut body_ui = ui.new_child(egui::UiBuilder::new().max_rect(body_rect));
        body_ui.add_space(16.0);

        draw_stat_row(
            &mut body_ui,
            &self.assets.closeness_icon,
            "Closeness",
            self.state.closeness,
            egui::Color32::from_rgb(120, 190, 230),
        );
        body_ui.add_space(14.0);
        draw_stat_row(
            &mut body_ui,
            &self.assets.stress_icon,
            "Stress",
            self.state.stress,
            egui::Color32::from_rgb(230, 120, 150),
        );
    }

    /// Hidden debug window: dumps the raw log — new outgoing JSON
    /// appended since the last send, and every raw pre-parse
    /// response/error string, in order. Doesn't auto-open; toggled with
    /// 'O'. Plain OS chrome since this is just for debugging.
    fn draw_raw_history_window(&mut self, ctx: &Context) {
        if !self.raw_history_open {
            return;
        }

        let viewport_id = egui::ViewportId::from_hash_of("raw_history_window");

        let builder = egui::ViewportBuilder::default()
            .with_title("raw_history")
            .with_inner_size([560.0, 600.0])
            .with_min_inner_size([300.0, 200.0])
            .with_decorations(true)
            .with_resizable(true)
            .with_transparent(false);

        let mut should_close = false;

        ctx.show_viewport_immediate(viewport_id, builder, |ctx, _class| {
            ctx.request_repaint();

            if ctx.input(|i| i.viewport().close_requested()) {
                should_close = true;
            }

            egui::CentralPanel::default().show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Copy All").clicked() {
                        let all = self.raw_log.join("\n\n");
                        ui.ctx().copy_text(all);
                    }
                    if ui.button("Clear").clicked() {
                        self.raw_log.clear();
                        // Next send will re-log the full context, since
                        // nothing is considered "already logged" anymore.
                        self.logged_message_count = 0;
                    }
                    ui.label(
                        egui::RichText::new(format!(
                            "{} entries (click-drag text below to select)",
                            self.raw_log.len()
                        ))
                        .weak()
                        .size(11.0),
                    );
                });
                ui.separator();

                egui::ScrollArea::vertical()
                    .id_salt("raw_history_scroll")
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if self.raw_log.is_empty() {
                            ui.label(
                                egui::RichText::new(
                                    "(nothing logged yet — send a message)",
                                )
                                .monospace()
                                .weak(),
                            );
                        }

                        for entry in &self.raw_log {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(entry).monospace().size(11.0),
                                )
                                .wrap_mode(egui::TextWrapMode::Wrap)
                                .selectable(true),
                            );
                            ui.add_space(4.0);
                            ui.separator();
                            ui.add_space(4.0);
                        }
                    });
            });
        });

        if should_close {
            self.raw_history_open = false;
        }
    }
}

/// Empty/whitespace-only proxy falls back to the default local Ollama
/// endpoint. A trailing slash is trimmed so `/api/chat` concatenation
/// doesn't end up with a double slash.
fn effective_base_url(proxy: &str) -> String {
    let trimmed = proxy.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        DEFAULT_BASE_URL.to_string()
    } else {
        trimmed.to_string()
    }
}

/// Shared hand-painted titlebar for the mini chrome-less windows (chat,
/// task manager, ...). `salt` keeps widget ids unique per-window.
/// Returns the titlebar's rect so callers know where the body starts.
fn draw_mini_titlebar(
    ui: &mut egui::Ui,
    ctx: &Context,
    rect: egui::Rect,
    salt: &str,
    title: &str,
    should_close: &mut bool,
) -> egui::Rect {
    let title_bar_height = 28.0;
    let title_bar_rect =
        egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), title_bar_height));

    let title_response = ui.interact(
        title_bar_rect,
        ui.id().with((salt, "titlebar")),
        egui::Sense::click_and_drag(),
    );

    ui.painter()
        .rect_filled(title_bar_rect, 0.0, egui::Color32::from_rgb(28, 36, 66));

    // Little icon chip, like the heart-eyes avatar chip in the reference.
    ui.painter().rect_filled(
        egui::Rect::from_min_size(
            title_bar_rect.min + egui::vec2(6.0, 5.0),
            egui::vec2(18.0, 18.0),
        ),
        3.0,
        egui::Color32::from_rgb(220, 70, 100),
    );

    ui.painter().text(
        title_bar_rect.min + egui::vec2(32.0, title_bar_height * 0.5),
        egui::Align2::LEFT_CENTER,
        title,
        egui::FontId::monospace(14.0),
        egui::Color32::WHITE,
    );

    // Minimize / maximize / close, right-aligned, beveled-button style.
    let button_size = egui::vec2(20.0, 18.0);
    let mut bx = title_bar_rect.max.x - 6.0;

    for (label, base_color) in [
        ("x", egui::Color32::from_rgb(205, 60, 70)),
        ("\u{25A1}", egui::Color32::from_rgb(70, 110, 170)), // □
        ("_", egui::Color32::from_rgb(70, 110, 170)),
    ] {
        bx -= button_size.x;
        let btn_rect =
            egui::Rect::from_min_size(egui::pos2(bx, title_bar_rect.min.y + 5.0), button_size);
        let resp = ui.interact(
            btn_rect,
            ui.id().with((salt, "titlebar_btn", label)),
            egui::Sense::click(),
        );

        let color = if resp.hovered() {
            base_color.gamma_multiply(1.3)
        } else {
            base_color
        };
        ui.painter().rect_filled(btn_rect, 3.0, color);
        ui.painter().text(
            btn_rect.center(),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::monospace(12.0),
            egui::Color32::WHITE,
        );

        if resp.clicked() && label == "x" {
            *should_close = true;
        }
        if resp.clicked() && label == "_" {
            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
        }

        bx -= 4.0;
    }

    if title_response.dragged() {
        ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
    }

    title_bar_rect
}

/// One icon + label + value + progress bar row, à la the reference
/// Task Manager screenshot.
fn draw_stat_row(
    ui: &mut egui::Ui,
    icon: &egui::TextureHandle,
    label: &str,
    value: f32,
    bar_color: egui::Color32,
) {
    ui.horizontal(|ui| {
        ui.add(
            egui::Image::new(icon)
                .texture_options(egui::TextureOptions::NEAREST)
                .fit_to_exact_size(egui::vec2(40.0, 40.0)),
        );

        ui.vertical(|ui| {
            ui.label(
                egui::RichText::new(label)
                    .color(egui::Color32::from_rgb(90, 60, 140))
                    .size(14.0),
            );
            ui.label(
                egui::RichText::new(format!("{:.0}/100", value.clamp(0.0, 100.0)))
                    .color(egui::Color32::from_rgb(70, 40, 130))
                    .strong()
                    .size(20.0),
            );

            let (response, painter) =
                ui.allocate_painter(egui::vec2(140.0, 14.0), egui::Sense::hover());
            let bar_rect = response.rect;

            painter.rect_filled(bar_rect, 3.0, egui::Color32::from_rgb(215, 205, 235));

            let filled_width = bar_rect.width() * (value.clamp(0.0, 100.0) / 100.0);
            let filled_rect = egui::Rect::from_min_size(
                bar_rect.min,
                egui::vec2(filled_width, bar_rect.height()),
            );
            painter.rect_filled(filled_rect, 3.0, bar_color);
        });
    });
}

/// How long to hold before revealing the next bubble in a multi-bubble
/// burst — roughly proportional to message length, like typing speed.
fn typing_delay(text: &str) -> f32 {
    (0.5 + text.chars().count() as f32 * 0.025).clamp(0.9, 3.2)
}

/// Ephemeral system note giving the model the real current time/day.
/// Rebuilt fresh on every send and appended to the outgoing messages
/// only — never pushed into `self.conversation` — so it's always
/// accurate and doesn't leave a stale timestamp sitting in history.
fn current_time_context() -> String {
    let now = chrono::Local::now();
    let part_of_day = match now.hour() {
        5..=11 => "morning",
        12..=16 => "afternoon",
        17..=20 => "evening",
        _ => "late at night",
    };

    format!(
        "[System note: the real-world time right now is {} ({}), on {}. \
         Weave this in naturally only if it genuinely fits — noticing \
         it's late, commenting on a weekend, etc. Don't recite the time \
         like a clock, don't treat this as something the user said, and \
         never mention receiving this note.]",
        now.format("%I:%M %p"),
        part_of_day,
        now.format("%A")
    )
}

// Max width of a bubble's text content, not counting the frame's margin.
const BUBBLE_TEXT_WIDTH: f32 = 220.0;

/// Builds a wrapped, width-locked label. `ui.set_max_width` alone doesn't
/// force wrapping — a `Label`'s wrap behavior defaults to "extend" inside
/// horizontal/custom layouts (like the ones the bubbles live in), so text
/// just runs past the frame instead of breaking. `allocate_ui_with_layout`
/// gives the label its own real width-bounded, top-down child `Ui`, and
/// `wrap_mode` forces line breaking within it.
fn wrapped_bubble_text(ui: &mut egui::Ui, text: &str, color: egui::Color32) {
    ui.allocate_ui_with_layout(
        egui::vec2(BUBBLE_TEXT_WIDTH, 0.0),
        egui::Layout::top_down(egui::Align::LEFT),
        |ui| {
            ui.add(
                egui::Label::new(egui::RichText::new(text).color(color))
                    .wrap_mode(egui::TextWrapMode::Wrap),
            );
        },
    );
}

/// Left-aligned bubble with the pet's avatar, for assistant replies.
fn draw_assistant_bubble(ui: &mut egui::Ui, avatar: &egui::TextureHandle, text: &str) {
    ui.horizontal(|ui| {
        ui.add(
            egui::Image::new(avatar)
                .texture_options(egui::TextureOptions::NEAREST)
                .fit_to_exact_size(egui::vec2(36.0, 36.0))
                .corner_radius(18.0),
        );

        egui::Frame::default()
            .fill(egui::Color32::from_rgb(190, 225, 250))
            .corner_radius(10.0)
            .inner_margin(10.0)
            .show(ui, |ui| {
                wrapped_bubble_text(ui, text, egui::Color32::from_rgb(30, 30, 60));
            });
    });
}

/// Right-aligned bubble, no avatar, for the player's own messages.
fn draw_user_bubble(ui: &mut egui::Ui, text: &str) {
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
        egui::Frame::default()
            .fill(egui::Color32::from_rgb(140, 220, 90))
            .corner_radius(10.0)
            .inner_margin(10.0)
            .show(ui, |ui| {
                wrapped_bubble_text(ui, text, egui::Color32::from_rgb(20, 40, 10));
            });
    });
}

/// Left-aligned sticker with the pet's avatar, for Ame's LLM-picked
/// stickers. No text frame/background — the sticker image is the
/// whole bubble.
fn draw_assistant_sticker(
    ui: &mut egui::Ui,
    avatar: &egui::TextureHandle,
    sticker: &egui::TextureHandle,
) {
    ui.horizontal(|ui| {
        ui.add(
            egui::Image::new(avatar)
                .texture_options(egui::TextureOptions::NEAREST)
                .fit_to_exact_size(egui::vec2(36.0, 36.0))
                .corner_radius(18.0),
        );
        ui.add(
            egui::Image::new(sticker)
                .texture_options(egui::TextureOptions::NEAREST)
                .fit_to_exact_size(egui::vec2(STICKER_SIZE, STICKER_SIZE)),
        );
    });
}

/// Right-aligned sticker, no avatar, for pchan's manually-sent stickers.
fn draw_user_sticker(ui: &mut egui::Ui, sticker: &egui::TextureHandle) {
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
        ui.add(
            egui::Image::new(sticker)
                .texture_options(egui::TextureOptions::NEAREST)
                .fit_to_exact_size(egui::vec2(STICKER_SIZE, STICKER_SIZE)),
        );
    });
}

impl eframe::App for PetApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        let delta = ctx.input(|i| i.stable_dt);

        //
        // Intercept the OS window-close button: play `out` once, then
        // actually exit the process.
        //
        if ctx.input(|i| i.viewport().close_requested())
            && self.state.activity != Activity::Closing
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.state.start_closing(&self.assets.animations);
        }

        //
        // Update internal state
        //
        self.state.update(delta, &self.assets.animations);

        //
        // Pick up any finished/failed LLM request
        //
        self.poll_llm();

        //
        // Reveal any queued multi-bubble replies as their typing delay
        // elapses.
        //
        self.advance_pending_bubbles(delta);

        //
        // If the user's gone quiet long enough, have Ame speak up first.
        //
        if let Some(idle_minutes) = self.state.poll_initiate() {
            self.initiate_chat(idle_minutes);
        }

        //
        // Decide which animation should play.
        //
        self.player.play(self.state.current_animation());

        //
        // Advance animation.
        //
        self.player.update(delta, &self.assets.animations);

        //
        // Draw the pet window.
        //
        Renderer::draw(
            ctx,
            &self.assets,
            &self.player,
            self.state.reply.as_deref(),
        );

        //
        // Draw the separate "Jine"-style chat window (text in/out +
        // model picker + sticker picker). Lives in its own OS window,
        // not overlaid on the pet anymore.
        //
        self.draw_chat_window(ctx);

        //
        // Draw the "Task Manager" window (closeness/stress readout).
        //
        self.draw_task_manager_window(ctx);

        //
        // Draw the secret raw chat history window, if toggled on.
        //
        self.draw_raw_history_window(ctx);

        //
        // Draw the settings window, if toggled on.
        //
        self.draw_settings_window(ctx);

        //
        // Temporary debug controls.
        //
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Num1) {
                self.state.set_activity(Activity::Idle);
            }
            if i.key_pressed(egui::Key::Num2) {
                self.state.set_activity(Activity::PhoneChat);
            }
            if i.key_pressed(egui::Key::Num3) {
                self.state.set_mood(Mood::Worried, &self.assets.animations);
            }
            if i.key_pressed(egui::Key::Num4) {
                self.state.set_mood(Mood::Excited, &self.assets.animations);
            }
            if i.key_pressed(egui::Key::Num5) {
                self.state.set_mood(Mood::Disappointed, &self.assets.animations);
            }
            if i.key_pressed(egui::Key::Num6) {
                self.state.set_mood(Mood::PissedOff, &self.assets.animations);
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                self.state.add_closeness(5.0);
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                self.state.add_closeness(-5.0);
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                self.state.add_stress(5.0);
            }
            if i.key_pressed(egui::Key::ArrowLeft) {
                self.state.add_stress(-5.0);
            }
            // Chat window got closed via its X button? Bring it back.
            if i.key_pressed(egui::Key::C) {
                self.chat_window_open = !self.chat_window_open;
            }
            // Same for the Task Manager window.
            if i.key_pressed(egui::Key::T) {
                self.task_manager_open = !self.task_manager_open;
            }
            // Secret raw chat history debug window.
            if i.key_pressed(egui::Key::O) {
                self.raw_history_open = !self.raw_history_open;
            }
        });

        ctx.request_repaint();
    }
}