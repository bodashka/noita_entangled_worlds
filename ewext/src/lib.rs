use std::{
    cell::{LazyCell, RefCell},
    ffi::{c_int, c_void},
    sync::LazyLock,
};

use iced_x86::Mnemonic;
use lua_bindings::{lua_State, Lua51, LUA_GLOBALSINDEX};
use noita::{NoitaPixelRun, ParticleWorldState};

mod lua_bindings;

mod noita;

mod addr_grabber;

static LUA: LazyLock<Lua51> = LazyLock::new(|| unsafe {
    let lib = libloading::Library::new("./lua51.dll").expect("library to exist");
    Lua51::from_library(lib).expect("library to be lua")
});

thread_local! {
    static STATE: LazyCell<RefCell<ExtState>> = LazyCell::new(|| ExtState::default().into());
}

struct GrabbedGlobals {
    game_global: *const c_void,
    world_state_entity: *const c_void,
}

#[derive(Default)]
struct ExtState {
    particle_world_state: Option<ParticleWorldState>,
    globals: Option<GrabbedGlobals>,
}

// const EWEXT: [(&'static str, Function); 1] = [("testfn", None)];

unsafe extern "C" fn init_particle_world_state(lua: *mut lua_State) -> c_int {
    println!("\nInitializing particle world state");
    let world_pointer = unsafe { LUA.lua_tointeger(lua, 1) };
    let chunk_map_pointer = unsafe { LUA.lua_tointeger(lua, 2) };
    let material_list_pointer = unsafe { LUA.lua_tointeger(lua, 3) };
    println!("pws stuff: {world_pointer:?} {chunk_map_pointer:?}");

    STATE.with(|state| {
        state.borrow_mut().particle_world_state = Some(ParticleWorldState {
            _world_ptr: world_pointer as *mut c_void,
            chunk_map_ptr: chunk_map_pointer as *mut c_void,
            material_list_ptr: material_list_pointer as _,
            runner: Default::default(),
        });
    });
    0
}

unsafe extern "C" fn encode_area(lua: *mut lua_State) -> c_int {
    let start_x = unsafe { LUA.lua_tointeger(lua, 1) } as i32;
    let start_y = unsafe { LUA.lua_tointeger(lua, 2) } as i32;
    let end_x = unsafe { LUA.lua_tointeger(lua, 3) } as i32;
    let end_y = unsafe { LUA.lua_tointeger(lua, 4) } as i32;
    let encoded_buffer = unsafe { LUA.lua_tointeger(lua, 5) } as *mut NoitaPixelRun;

    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let pws = state.particle_world_state.as_mut().unwrap();
        let runs = unsafe { pws.encode_area(start_x, start_y, end_x, end_y, encoded_buffer) };
        unsafe { LUA.lua_pushinteger(lua, runs as isize) };
    });
    1
}

unsafe fn grab_addrs(lua: *mut lua_State) {
    LUA.lua_getfield(lua, LUA_GLOBALSINDEX, c"GameGetWorldStateEntity".as_ptr());
    let base = LUA.lua_tocfunction(lua, -1).unwrap() as *const c_void;
    let world_state_entity =
        addr_grabber::grab_addr_from_instruction(base, 0x007aa7ce - 0x007aa540, Mnemonic::Mov);
    println!(
        "World state entity addr: 0x{:x}",
        world_state_entity as usize
    );
    // Pop the last element.
    LUA.lua_settop(lua, -2);

    LUA.lua_getfield(lua, LUA_GLOBALSINDEX, c"GameGetFrameNum".as_ptr());
    let base = LUA.lua_tocfunction(lua, -1).unwrap() as *const c_void;
    let load_game_global =
        addr_grabber::grab_addr_from_instruction(base, 0x007bf3c9 - 0x007bf140, Mnemonic::Call); // CALL load_game_global
    println!("Load game global addr: 0x{:x}", load_game_global as usize);
    let game_global = addr_grabber::grab_addr_from_instruction(
        load_game_global,
        0x00439c17 - 0x00439bb0,
        Mnemonic::Mov,
    );
    println!("Game global addr: 0x{:x}", game_global as usize);
    // Pop the last element.
    LUA.lua_settop(lua, -2);

    STATE.with(|state| {
        state.borrow_mut().globals = Some(GrabbedGlobals {
            game_global,
            world_state_entity,
        });
    });
}

/// # Safety
///
/// Only gets called by lua when loading a module.
#[no_mangle]
pub unsafe extern "C" fn luaopen_ewext0(lua: *mut lua_State) -> c_int {
    println!("Initializing ewext");
    unsafe {
        grab_addrs(lua);

        LUA.lua_createtable(lua, 0, 0);

        LUA.lua_pushcclosure(lua, Some(init_particle_world_state), 0);
        LUA.lua_setfield(lua, -2, c"init_particle_world_state".as_ptr());
        LUA.lua_pushcclosure(lua, Some(encode_area), 0);
        LUA.lua_setfield(lua, -2, c"encode_area".as_ptr());
    }
    println!("Initializing ewext - Ok");
    1
}
