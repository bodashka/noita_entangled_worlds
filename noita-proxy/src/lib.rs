use std::{
    fmt::Display,
    net::SocketAddr,
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use bitcode::{Decode, Encode};
use clipboard::{ClipboardContext, ClipboardProvider};
use eframe::egui::{self, Align2, Color32, InnerResponse, Margin, RichText, TextureOptions, Ui};
use mod_manager::{Modmanager, ModmanagerSettings};
use net::{omni::PeerVariant, NetManagerInit};
use self_update::SelfUpdateManager;
use serde::{Deserialize, Serialize};
use steamworks::{LobbyId, SteamAPIInitError};
use tangled::Peer;
use tracing::info;

pub mod messages;
mod mod_manager;
pub mod net;
pub mod releases;
mod self_update;
pub mod steam_helper;

#[derive(Debug, Decode, Encode, Clone)]
pub struct GameSettings {
    seed: u64,
    debug_mode: bool,
    world_sync_version: u32,
}

enum AppState {
    Connect,
    ModManager,
    Netman { netman: Arc<net::NetManager> },
    Error { message: String },
    SelfUpdate,
}

#[derive(Debug, Serialize, Deserialize)]
struct AppSavedState {
    addr: String,
    debug_mode: bool,
    use_constant_seed: bool,
    nickname: Option<String>,
    times_started: u32,
    world_sync_version: u32,
}

impl Default for AppSavedState {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:5123".to_string(),
            debug_mode: false,
            use_constant_seed: false,
            nickname: None,
            times_started: 0,
            world_sync_version: 1,
        }
    }
}

pub struct App {
    state: AppState,
    modmanager: Modmanager,
    steam_state: Result<steam_helper::SteamState, SteamAPIInitError>,
    saved_state: AppSavedState,
    modmanager_settings: ModmanagerSettings,
    self_update: SelfUpdateManager,
}

const MODMANAGER: &str = "modman";

fn filled_group<R>(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui) -> R) -> InnerResponse<R> {
    let style = ui.style();
    let frame = egui::Frame {
        inner_margin: Margin::same(6.0), // same and symmetric looks best in corners when nesting groups
        rounding: style.visuals.widgets.noninteractive.rounding,
        stroke: style.visuals.widgets.noninteractive.bg_stroke,
        fill: Color32::from_rgba_premultiplied(20, 20, 20, 180),
        ..Default::default()
    };
    frame.show(ui, add_contents)
}

fn heading_with_underline(ui: &mut Ui, text: impl Into<RichText>) {
    ui.vertical_centered_justified(|ui| {
        ui.heading(text);
    });
    ui.separator();
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut saved_state: AppSavedState = cc
            .storage
            .and_then(|storage| eframe::get_value(storage, eframe::APP_KEY))
            .unwrap_or_default();
        let modmanager_settings = cc
            .storage
            .and_then(|storage| eframe::get_value(storage, MODMANAGER))
            .unwrap_or_default();
        saved_state.times_started += 1;
        egui_extras::install_image_loaders(&cc.egui_ctx);
        info!("Creating the app...");
        Self {
            state: AppState::ModManager,
            modmanager: Modmanager::default(),
            steam_state: steam_helper::SteamState::new(),
            saved_state,
            modmanager_settings,
            self_update: SelfUpdateManager::new(),
        }
    }

    fn get_netman_init(&self) -> NetManagerInit {
        let steam_nickname = if let Ok(steam) = &self.steam_state {
            Some(steam.get_user_name(steam.get_my_id()))
        } else {
            None
        };
        let my_nickname = self.saved_state.nickname.clone().or(steam_nickname);
        NetManagerInit { my_nickname }
    }

    fn start_server(&mut self) {
        let bind_addr = "0.0.0.0:5123".parse().unwrap();
        let peer = Peer::host(bind_addr, None).unwrap();
        let netman = net::NetManager::new(PeerVariant::Tangled(peer), self.get_netman_init());
        self.set_netman_settings(&netman);
        netman.clone().start();
        self.state = AppState::Netman { netman };
    }

    fn set_netman_settings(&mut self, netman: &Arc<net::NetManager>) {
        let mut settings = netman.settings.lock().unwrap();
        settings.debug_mode = self.saved_state.debug_mode;
        settings.world_sync_version = self.saved_state.world_sync_version;
        if !self.saved_state.use_constant_seed {
            settings.seed = rand::random();
        }
        netman.accept_local.store(true, Ordering::SeqCst);
    }
    fn start_connect(&mut self, addr: SocketAddr) {
        let peer = Peer::connect(addr, None).unwrap();
        let netman = net::NetManager::new(PeerVariant::Tangled(peer), self.get_netman_init());
        netman.clone().start();
        self.state = AppState::Netman { netman };
    }

    fn start_steam_host(&mut self) {
        let peer = net::steam_networking::SteamPeer::new_host(
            steamworks::LobbyType::Private,
            self.steam_state.as_ref().unwrap().client.clone(),
        );
        let netman = net::NetManager::new(PeerVariant::Steam(peer), self.get_netman_init());
        self.set_netman_settings(&netman);
        netman.clone().start();
        self.state = AppState::Netman { netman };
    }

    fn notify_error(&mut self, error: impl Display) {
        self.state = AppState::Error {
            message: error.to_string(),
        }
    }

    fn start_steam_connect(&mut self, id: LobbyId) {
        let peer = net::steam_networking::SteamPeer::new_connect(
            id,
            self.steam_state.as_ref().unwrap().client.clone(),
        );
        let netman = net::NetManager::new(PeerVariant::Steam(peer), self.get_netman_init());
        netman.clone().start();
        self.state = AppState::Netman { netman };
    }

    fn connect_screen(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("version_panel")
            .exact_height(25.0)
            .show(ctx, |ui| {
                self.self_update.display_version(ui);

                if self.self_update.request_update {
                    self.state = AppState::SelfUpdate;
                }
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.saved_state.times_started % 5 == 0 {
                let image = egui::Image::new(egui::include_image!("../assets/longleg.png"))
                    .texture_options(TextureOptions::NEAREST)
                    .maintain_aspect_ratio(true);
                image.paint_at(ui, ui.ctx().screen_rect());
            }

            let item_spacing = ui.spacing().item_spacing.x;
            let rect = ui.max_rect();
            let (settings_rect, right) = rect.split_left_right_at_fraction(0.5);
            let (steam_connect_rect, ip_connect_rect) = right.split_top_bottom_at_fraction(0.5);
            ui.allocate_ui_at_rect(settings_rect.shrink(item_spacing), |ui| {
                filled_group(ui, |ui| {
                    ui.set_min_size(ui.available_size());
                    heading_with_underline(ui, "Game settings");

                    ui.label("Debug settings");
                    ui.checkbox(&mut self.saved_state.debug_mode, "Debug/cheat mode");
                    ui.checkbox(&mut self.saved_state.use_constant_seed, "Use fixed seed");

                    ui.add_space(20.0);

                    ui.label("World sync version to use:");
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.saved_state.world_sync_version, 1, "v1");
                        ui.radio_value(
                            &mut self.saved_state.world_sync_version,
                            2,
                            "v2 (experimental)",
                        );
                    });
                });
            });
            ui.allocate_ui_at_rect(steam_connect_rect.shrink(item_spacing), |ui| {
                filled_group(ui, |ui| {
                    ui.set_min_size(ui.available_size());

                    heading_with_underline(ui, "Connect using steam");

                    match &self.steam_state {
                        Ok(_) => {
                            if ui.button("Create lobby").clicked() {
                                self.start_steam_host();
                            }
                            if ui.button("Connect to lobby in clipboard").clicked() {
                                let id = ClipboardProvider::new()
                                    .and_then(|mut ctx: ClipboardContext| ctx.get_contents());
                                match id {
                                    Ok(id) => {
                                        let id = id.parse().map(LobbyId::from_raw);
                                        match id {
                                            Ok(id) => self.start_steam_connect(id),
                                            Err(error) => self.notify_error(error),
                                        }
                                    }
                                    Err(error) => self.notify_error(error),
                                }
                            }
                        }
                        Err(err) => {
                            ui.label(format!("Could not init steam networking: {}", err));
                        }
                    }
                });
            });
            ui.allocate_ui_at_rect(ip_connect_rect.shrink(item_spacing), |ui| {
                filled_group(ui, |ui| {
                    ui.set_min_size(ui.available_size());

                    heading_with_underline(ui, "Connect by ip");

                    ui.label("Note: steam networking is more reliable. Use it, if possible.");
                    if ui.button("Host").clicked() {
                        self.start_server();
                    }

                    ui.text_edit_singleline(&mut self.saved_state.addr);
                    let addr = self.saved_state.addr.parse();
                    ui.add_enabled_ui(addr.is_ok(), |ui| {
                        if ui.button("Connect").clicked() {
                            if let Ok(addr) = addr {
                                self.start_connect(addr);
                            }
                        }
                    });
                });
            });
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_secs(1));
        match &self.state {
            AppState::Connect => {
                self.connect_screen(ctx);
            }
            AppState::Netman { netman } => {
                let stopped = netman.stopped.load(Ordering::Relaxed);
                let accept_local = netman.accept_local.load(Ordering::Relaxed);
                let local_connected = netman.local_connected.load(Ordering::Relaxed);
                egui::TopBottomPanel::top("noita_status").show(ctx, |ui| {
                    ui.add_space(3.0);
                    if accept_local {
                        if local_connected {
                            ui.colored_label(Color32::GREEN, "Local Noita instance connected");
                        } else {
                            ui.colored_label(Color32::YELLOW, "Awaiting Noita connection. It's time to start new game in Noita now!");
                        }
                    } else {
                        ui.label("Not yet ready");
                    }
                });
                egui::SidePanel::left("players")
                    .resizable(false)
                    .exact_width(200.0)
                    .show(ctx, |ui| {
                        ui.add_space(3.0);
                        if netman.peer.is_steam() {
                            let steam = self.steam_state.as_mut().expect(
                                "steam should be available, as we are using steam networking",
                            );
                            for peer in netman.peer.iter_peer_ids() {
                                let role = peer_role(peer, netman);

                                let username = steam.get_user_name(peer.into());
                                let avatar = steam.get_avatar(ctx, peer.into());
                                if let Some(avatar) = avatar {
                                    avatar.display_with_labels(ui, &username, role);
                                    ui.add_space(5.0);
                                } else {
                                    ui.label(&username);
                                }
                            }
                        } else {
                            for peer in netman.peer.iter_peer_ids() {
                                ui.label(peer.to_string());
                            }
                        }
                    });
                egui::CentralPanel::default().show(ctx, |ui| {
                    if stopped {
                        ui.colored_label(Color32::LIGHT_RED, "Netmanager thread has stopped");
                        if let Some(err) = netman.error.lock().unwrap().as_ref() {
                            ui.label("With the following error:");
                            ui.label(err.to_string());
                        }
                        ui.separator();
                    }

                    if netman.peer.is_steam() {
                        if let Some(id) = netman.peer.lobby_id() {
                            if ui.button("Save lobby id to clipboard").clicked() {
                                let mut ctx: ClipboardContext = ClipboardProvider::new().unwrap();
                                let _ = ctx.set_contents(id.raw().to_string());
                            }
                        } else {
                            ui.label("Lobby id not available");
                        }
                    }
                    ui.label(format!("Peer state: {}", netman.peer.state()));
                });
            }
            AppState::Error { message } => {
                if egui::CentralPanel::default()
                    .show(ctx, |ui| {
                        ui.heading("An error occured:");
                        ui.label(message);
                        ui.button("Back").clicked()
                    })
                    .inner
                {
                    self.state = AppState::Connect;
                }
            }
            AppState::ModManager => {
                egui::Window::new("Mod manager")
                    .auto_sized()
                    .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        self.modmanager.update(
                            ctx,
                            ui,
                            &mut self.modmanager_settings,
                            self.steam_state.as_mut().ok(),
                        )
                    });
                if self.modmanager.is_done() {
                    self.state = AppState::Connect;
                }
            }
            AppState::SelfUpdate => {
                egui::Window::new("Self update")
                    .auto_sized()
                    .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        self.self_update.self_update(ui);
                    });
            }
        };
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.saved_state);
        eframe::set_value(storage, MODMANAGER, &self.modmanager_settings);
    }
}

fn peer_role(peer: net::omni::OmniPeerId, netman: &Arc<net::NetManager>) -> &str {
    if peer == netman.peer.host_id() {
        "Host"
    } else {
        if Some(peer) == netman.peer.my_id() {
            "Me"
        } else {
            "Player"
        }
    }
}
