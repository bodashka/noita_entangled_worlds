// Define Tile and Spell structs for flexibility and pathfinding
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tile {
    Solid,
    Empty,
    Liquid,
    Dangerous,
}

#[derive(Debug)]
struct Position {
    x: usize,
    y: usize,
}

#[derive(Debug, Clone)]
struct Spell {
    id: u32,
    range: usize,
    damage: f32,
    cooldown: usize,
    duration: usize,
    cost: f32,  // Resource cost (like mana or stamina)
    spell_type: String, // e.g., "teleport", "heal", "damage"
}

#[derive(Debug, Clone)]
struct Wand {
    spells: Vec<Spell>,  // Ordered list of spells
}

#[derive(Debug)]
struct Agent {
    position: Position,
    health: f32,
    oxygen: f32,
    levitation: f32,
    wands: Vec<Wand>,
}

#[derive(Debug)]
struct Environment {
    grid: Vec<Vec<Tile>>,
    agent: Agent,
}