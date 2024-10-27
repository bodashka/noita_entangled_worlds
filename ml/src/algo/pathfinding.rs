use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

// Existing imports from previous code omitted for brevity...

#[derive(Clone, PartialEq, Eq)]
struct Node {
    position: Position,
    cost: f32,
    priority: f32,
    levitation: f32,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.partial_cmp(&self.priority).unwrap()
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Node {
    fn new(position: Position, cost: f32, priority: f32, levitation: f32) -> Self {
        Node { position, cost, priority, levitation }
    }
}

impl Environment {
    fn tile_cost(&self, tile: Tile, levitation: f32) -> f32 {
        match tile {
            Tile::Solid => f32::INFINITY,  // Cannot pass through solid
            Tile::Empty => 1.0,            // Base cost for empty tiles
            Tile::Liquid => 3.0,           // Higher cost for liquid due to oxygen
            Tile::Dangerous => 5.0,        // Dangerous tiles have a high cost
        } + if levitation > 0.0 { 0.5 } else { 1.0 }  // Additional levitation cost
    }

    fn heuristic(&self, current: &Position, target: &Position) -> f32 {
        let dx = (current.x as isize - target.x as isize).abs() as f32;
        let dy = (current.y as isize - target.y as isize).abs() as f32;
        dx + dy
    }

    fn find_path(&mut self, start: Position, target: Position) -> Option<Vec<Position>> {
        let mut open_set = BinaryHeap::new();
        let mut came_from: HashMap<Position, Position> = HashMap::new();
        let mut g_score: HashMap<Position, f32> = HashMap::new();
        
        g_score.insert(start, 0.0);
        
        let start_node = Node::new(start, 0.0, self.heuristic(&start, &target), self.agent.levitation);
        open_set.push(start_node);

        while let Some(current) = open_set.pop() {
            if current.position == target {
                return Some(self.reconstruct_path(came_from, current.position));
            }

            let neighbors = self.get_neighbors(&current.position);
            for (neighbor_pos, tile) in neighbors {
                let levitation_cost = if neighbor_pos.y > current.position.y { 1.0 } else { 0.0 };
                let tentative_g_score = g_score[&current.position] + self.tile_cost(tile, current.levitation - levitation_cost);
                
                if tentative_g_score < *g_score.get(&neighbor_pos).unwrap_or(&f32::INFINITY) {
                    came_from.insert(neighbor_pos, current.position);
                    g_score.insert(neighbor_pos, tentative_g_score);
                    let priority = tentative_g_score + self.heuristic(&neighbor_pos, &target);

                    if current.levitation > levitation_cost {
                        open_set.push(Node::new(neighbor_pos, tentative_g_score, priority, current.levitation - levitation_cost));
                    }
                }
            }
        }
        None
    }

    fn get_neighbors(&self, pos: &Position) -> Vec<(Position, Tile)> {
        let mut neighbors = Vec::new();

        let potential_moves = vec![
            (pos.x.wrapping_sub(1), pos.y),   // Left
            (pos.x + 1, pos.y),               // Right
            (pos.x, pos.y.wrapping_sub(1)),   // Down
            (pos.x, pos.y + 1),               // Up
        ];

        for (x, y) in potential_moves {
            if x < GRID_SIZE && y < GRID_SIZE {
                neighbors.push((Position { x, y }, self.grid[x][y]));
            }
        }
        neighbors
    }

    fn reconstruct_path(&self, came_from: HashMap<Position, Position>, current: Position) -> Vec<Position> {
        let mut total_path = vec![current];
        let mut curr = current;
        while let Some(&prev) = came_from.get(&curr) {
            total_path.push(prev);
            curr = prev;
        }
        total_path.reverse();
        total_path
    }
}

// Main function with training and pathfinding demonstration
fn main() {
    let mut env = Environment::new();
    let agent = RLAgent::new(Shape::new([VISIBLE_AREA * VISIBLE_AREA]));
    let optimizer = Adam::new(1e-4);

    // Set a target position to demonstrate pathfinding
    let target = Position { x: 50, y: 50 };
    
    // Perform pathfinding
    if let Some(path) = env.find_path(env.agent.position, target) {
        println!("Path found: {:?}", path);
    } else {
        println!("No path found.");
    }

    // Train the agent
    train_agent(&mut env, &agent, &optimizer);
}
