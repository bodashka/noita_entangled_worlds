use std::path::PathBuf;

use argh::FromArgs;

#[derive(FromArgs, PartialEq, Debug)]
/// Noita proxy.
pub struct Args {
    /// noita launch command that will be used.
    #[argh(option)]
    pub launch_cmd: Option<String>,
    /// adjust ui scale; default is 1.0.
    #[argh(option)]
    pub ui_zoom_factor: Option<f32>,
    /// path to crashcatcher's replay folder.
    #[argh(option)]
    pub replay_folder: Option<PathBuf>,
}
