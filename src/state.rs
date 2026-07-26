use std::collections::HashMap;

use chrono::Timelike;

use crate::animation::AnimationClip;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relationship {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdleMood {
    Normal,
    Anxiety,
    Iraira,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReactiveKind {
    Positive,
    Negative,
    Pouting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activity {
    Idle,
    PhoneChat,
    Reactive(ReactiveKind),
    Sleeping,
    Closing,
}

/// Mood string returned by the LLM's structured JSON reply:
/// `{"reply": "...", "mood": "..."}`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mood {
    Worried,
    Excited,
    Disappointed,
    PissedOff,
    Neutral,
}

impl Mood {
    pub fn from_llm(raw: &str) -> Self {
        match raw.trim().to_lowercase().as_str() {
            "worried" => Mood::Worried,
            "excited" => Mood::Excited,
            "disappointed" => Mood::Disappointed,
            "pissed_off" | "pissed off" | "angry" => Mood::PissedOff,
            _ => Mood::Neutral,
        }
    }
}

pub struct PetState {
    pub closeness: f32,
    pub stress: f32,
    pub relationship: Relationship,
    pub activity: Activity,
    pub idle_mood: IdleMood,

    /// Counts down while a one-shot clip (reactive/sleep/closing) is
    /// playing. When it hits zero, `finish_one_shot` fires.
    one_shot_timer: f32,

    /// Seconds until the next "should I show the sleep overlay?" roll.
    sleep_countdown: f32,

    /// Latest LLM reply to show in a speech bubble, and how much longer
    /// (in seconds) it should stay on screen.
    pub reply: Option<String>,
    pub reply_timer: f32,

    /// Seconds since the last *real* user message. Drives both the AFK
    /// closeness/stress penalty and the "Ame initiates chat" scheduler.
    idle_seconds: f32,
    /// idle_seconds value at which the next unprompted "hey, you there?"
    /// ping should fire.
    next_initiate_at: f32,
    /// How many 2-hour AFK penalty blocks have already been applied for
    /// the current idle stretch.
    afk_penalty_ticks: u32,
}

impl PetState {
    pub fn new() -> Self {
        Self {
            closeness: 50.0,
            stress: 0.0,
            relationship: Relationship::B,
            activity: Activity::Idle,
            idle_mood: IdleMood::Normal,
            one_shot_timer: 0.0,
            sleep_countdown: pseudo_random_range(300.0, 1800.0), // 5-30 min
            reply: None,
            reply_timer: 0.0,

            idle_seconds: 0.0,
            next_initiate_at: pseudo_random_range(120.0, 3000.0), // 2-50 min
            afk_penalty_ticks: 0,
        }
    }

    pub fn update(&mut self, delta: f32, animations: &HashMap<String, AnimationClip>) {
        self.relationship = self.calculate_relationship();

        if self.reply_timer > 0.0 {
            self.reply_timer -= delta;
            if self.reply_timer <= 0.0 {
                self.reply = None;
            }
        }

        if self.one_shot_timer > 0.0 {
            self.one_shot_timer -= delta;
            if self.one_shot_timer <= 0.0 {
                self.finish_one_shot();
            }
        }

        // AFK neglect: every full 2-hour block since the last real
        // message costs closeness and adds stress. Can fire more than
        // once if the user's been gone a long time.
        self.idle_seconds += delta;
        let elapsed_afk_blocks = (self.idle_seconds / 7200.0) as u32;
        while self.afk_penalty_ticks < elapsed_afk_blocks {
            self.add_closeness(-5.0);
            self.add_stress(3.0);
            self.afk_penalty_ticks += 1;
        }

        // Random sleep overlay: only while genuinely idle, and only late
        // at night. Interrupting chat/reactions is explicitly excluded.
        if self.activity == Activity::Idle && is_late_night() {
            self.sleep_countdown -= delta;
            if self.sleep_countdown <= 0.0 {
                self.start_sleep(animations);
            }
        }
    }

    /// Call this whenever the player actually sends a chat message.
    /// Nudges closeness up / stress down and resets the AFK clock.
    pub fn record_chat_interaction(&mut self) {
        self.add_closeness(0.4);
        self.add_stress(-0.1);
        self.idle_seconds = 0.0;
        self.afk_penalty_ticks = 0;
        self.next_initiate_at = pseudo_random_range(120.0, 3000.0);
    }

    /// Poll every frame from the app layer. Returns `Some(idle_minutes)`
    /// once the current random 2-50 minute silence window has elapsed,
    /// then rolls the next window. Doesn't reset `idle_seconds`, so AFK
    /// neglect keeps accumulating even while Ame is trying to start
    /// conversations on her own.
    pub fn poll_initiate(&mut self) -> Option<f32> {
        if self.idle_seconds >= self.next_initiate_at {
            let idle_minutes = self.idle_seconds / 60.0;
            self.next_initiate_at = self.idle_seconds + pseudo_random_range(120.0, 3000.0);
            Some(idle_minutes)
        } else {
            None
        }
    }

    fn calculate_relationship(&self) -> Relationship {
        // Stress has priority.
        if self.stress >= 80.0 {
            return Relationship::E;
        }
        if self.stress >= 50.0 {
            return Relationship::F;
        }

        // Debug-only override: at zero stress and very high closeness,
        // show off the G idle instead of topping out at C.
        #[cfg(debug_assertions)]
        {
            if self.stress == 0.0 && self.closeness > 85.0 {
                return Relationship::G;
            }
        }

        // Normal progression, low -> high: D, G, A, B, C.
        if self.closeness < 10.0 {
            return Relationship::D;
        }
        if self.closeness < 20.0 {
            return Relationship::G;
        }
        if self.closeness < 40.0 {
            return Relationship::A;
        }
        if self.closeness < 60.0 {
            return Relationship::B;
        }
        Relationship::C
    }

    fn tier(&self) -> &'static str {
        match self.relationship {
            Relationship::A => "a",
            Relationship::B => "b",
            Relationship::C => "c",
            Relationship::D => "d",
            Relationship::E => "e",
            Relationship::F => "f",
            Relationship::G => "g",
        }
    }

    pub fn idle_animation(&self) -> &'static str {
        match self.relationship {
            Relationship::A => "idle_normal.normal_a",
            Relationship::B => "idle_normal.normal_b",
            Relationship::C => "idle_normal.normal_c",
            Relationship::D => "idle_normal.normal_d",
            Relationship::E => "idle_normal.normal_e",
            Relationship::F => "idle_normal.normal_f",
            Relationship::G => "idle_normal.normal_g",
        }
    }

    pub fn anxiety_animation(&self) -> &'static str {
        match self.relationship {
            Relationship::A => "idle_anxiety.anxiety_a",
            Relationship::B => "idle_anxiety.anxiety_b",
            Relationship::C => "idle_anxiety.anxiety_c",
            Relationship::D => "idle_anxiety.anxiety_d",
            Relationship::E => "idle_anxiety.anxiety_e",
            Relationship::F => "idle_anxiety.anxiety_f",
            Relationship::G => "idle_anxiety.anxiety_g",
        }
    }

    pub fn iraira_animation(&self) -> &'static str {
        match self.relationship {
            Relationship::A => "idle_iraira.iraira_a",
            Relationship::B => "idle_iraira.iraira_b",
            Relationship::C => "idle_iraira.iraira_c",
            Relationship::D => "idle_iraira.iraira_d",
            Relationship::E => "idle_iraira.iraira_e",
            Relationship::F => "idle_iraira.iraira_f",
            Relationship::G => "idle_iraira.iraira_g",
        }
    }

    pub fn phone_animation(&self) -> &'static str {
        match self.relationship {
            Relationship::A => "phone_chat.talk_a",
            Relationship::B => "phone_chat.talk_b",
            Relationship::C => "phone_chat.talk_c",
            Relationship::D => "phone_chat.talk_d",
            Relationship::E => "phone_chat.talk_e",
            Relationship::F => "phone_chat.talk_f",
            Relationship::G => "phone_chat.talk_g",
        }
    }

    pub fn positive_animation(&self) -> &'static str {
        match self.relationship {
            Relationship::A => "positive.positive_a",
            Relationship::B => "positive.positive_b",
            Relationship::C => "positive.positive_c",
            Relationship::D => "positive.positive_d",
            Relationship::E => "positive.positive_e",
            Relationship::F => "positive.positive_f",
            Relationship::G => "positive.positive_g",
        }
    }

    pub fn negative_animation(&self) -> &'static str {
        match self.relationship {
            Relationship::A => "negative.negative_a",
            Relationship::B => "negative.negative_b",
            Relationship::C => "negative.negative_c",
            Relationship::D => "negative.negative_d",
            Relationship::E => "negative.negative_e",
            Relationship::F => "negative.negative_f",
            Relationship::G => "negative.negative_g",
        }
    }

    pub fn out_animation(&self) -> &'static str {
        match self.relationship {
            Relationship::A => "out.out_a",
            Relationship::B => "out.out_b",
            Relationship::C => "out.out_c",
            Relationship::D => "out.out_d",
            Relationship::E => "out.out_e",
            Relationship::F => "out.out_f",
            Relationship::G => "out.out_g",
        }
    }

    pub fn current_animation(&self) -> &'static str {
        match self.activity {
            Activity::Idle => match self.idle_mood {
                IdleMood::Normal => self.idle_animation(),
                IdleMood::Anxiety => self.anxiety_animation(),
                IdleMood::Iraira => self.iraira_animation(),
            },
            Activity::PhoneChat => self.phone_animation(),
            Activity::Reactive(ReactiveKind::Positive) => self.positive_animation(),
            Activity::Reactive(ReactiveKind::Negative) => self.negative_animation(),
            Activity::Reactive(ReactiveKind::Pouting) => "pouting",
            Activity::Sleeping => "sleep",
            Activity::Closing => self.out_animation(),
        }
    }

    pub fn set_activity(&mut self, activity: Activity) {
        self.activity = activity;
    }

    pub fn add_closeness(&mut self, amount: f32) {
        self.closeness = (self.closeness + amount).clamp(0.0, 100.0);
    }

    pub fn add_stress(&mut self, amount: f32) {
        self.stress = (self.stress + amount).clamp(0.0, 100.0);
    }

    /// Compact description of the current relationship state, meant to be
    /// injected into the LLM's prompt so replies stay in character.
    pub fn state_prompt(&self) -> String {
        let tier_desc = match self.relationship {
            Relationship::D => "You know each other yet; distant and a little dismissive.",
            Relationship::G => "You're starting to warm up to them.",
            Relationship::A => "You're becoming friendly and comfortable with them.",
            Relationship::B => "You're warm and familiar.",
            Relationship::C => "You're very close, affectionate and playful.",
            Relationship::E => "You're extremely stressed and overwhelmed right now.",
            Relationship::F => "You're stressed and a bit on edge right now.",
        };

        format!(
            "[Current state] closeness: {:.0}/100, stress: {:.0}/100. {}",
            self.closeness, self.stress, tier_desc
        )
    }

    /// Show a reply bubble for `seconds` seconds.
    pub fn show_reply(&mut self, text: impl Into<String>, seconds: f32) {
        self.reply = Some(text.into());
        self.reply_timer = seconds;
    }

    /// Apply the mood the LLM returned alongside its reply. Reactive
    /// moods (excited/disappointed/pissed_off) play their one-shot clip
    /// once, then settle into the matching idle. Worried/neutral just
    /// switch the idle loop directly.
    pub fn set_mood(&mut self, mood: Mood, animations: &HashMap<String, AnimationClip>) {
        match mood {
            Mood::Worried => {
                self.idle_mood = IdleMood::Anxiety;
                self.activity = Activity::Idle;
                self.one_shot_timer = 0.0;
            }
            Mood::Neutral => {
                self.idle_mood = IdleMood::Normal;
                self.activity = Activity::Idle;
                self.one_shot_timer = 0.0;
            }
            Mood::Excited => self.start_reactive(ReactiveKind::Positive, animations),
            Mood::Disappointed => self.start_reactive(ReactiveKind::Negative, animations),
            Mood::PissedOff => self.start_reactive(ReactiveKind::Pouting, animations),
        }
    }

    fn start_reactive(&mut self, kind: ReactiveKind, animations: &HashMap<String, AnimationClip>) {
        self.activity = Activity::Reactive(kind);
        let id = match kind {
            ReactiveKind::Positive => self.positive_animation(),
            ReactiveKind::Negative => self.negative_animation(),
            ReactiveKind::Pouting => "pouting",
        };
        self.one_shot_timer = clip_duration(animations, id);
    }

    fn start_sleep(&mut self, animations: &HashMap<String, AnimationClip>) {
        self.activity = Activity::Sleeping;
        self.one_shot_timer = clip_duration(animations, "sleep");
        self.sleep_countdown = pseudo_random_range(300.0, 1800.0); // 5-30 min
    }

    /// Called by the window-close handler to intercept the OS close
    /// button: play `out` once, then the app exits for real.
    pub fn start_closing(&mut self, animations: &HashMap<String, AnimationClip>) {
        if self.activity == Activity::Closing {
            return;
        }
        self.activity = Activity::Closing;
        let id = self.out_animation();
        self.one_shot_timer = clip_duration(animations, id);
    }

    fn finish_one_shot(&mut self) {
        match self.activity {
            Activity::Reactive(ReactiveKind::Positive) | Activity::Reactive(ReactiveKind::Negative) => {
                self.idle_mood = IdleMood::Normal;
                self.activity = Activity::Idle;
            }
            Activity::Reactive(ReactiveKind::Pouting) => {
                // "pouting then continue to iraira" — iraira then loops
                // indefinitely until the next trigger/event.
                self.idle_mood = IdleMood::Iraira;
                self.activity = Activity::Idle;
            }
            Activity::Sleeping => {
                self.activity = Activity::Idle;
            }
            Activity::Closing => {
                std::process::exit(0);
            }
            _ => {}
        }
    }
}

fn clip_duration(animations: &HashMap<String, AnimationClip>, id: &str) -> f32 {
    animations
        .get(id)
        .filter(|clip| !clip.frames.is_empty())
        .map(|clip| clip.frames.len() as f32 / clip.config.fps.max(0.01))
        .unwrap_or(1.0)
}

fn is_late_night() -> bool {
    chrono::Local::now().hour() >= 22
}

/// Tiny dependency-free RNG substitute — good enough for jittering a
/// timer, not for anything security-sensitive.
fn pseudo_random_range(min: f32, max: f32) -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let t = (nanos % 100_000) as f32 / 100_000.0;
    min + t * (max - min)
}