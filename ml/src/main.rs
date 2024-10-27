use burn::nn::{self, Linear, Module, ReLU, Conv2d, LSTM};
use burn::tensor::{self, Tensor, Shape};
use burn::optim::Adam;
use rand::Rng;

// Define grid size and agent's visible area for training
const GRID_SIZE: usize = 512;
const VISIBLE_AREA: usize = 20;  // Agent's visible area around itself
const MAX_LEVITATION: f32 = 100.0;

impl Environment {
    fn new() -> Self {
        let mut grid = vec![vec![Tile::Empty; GRID_SIZE]; GRID_SIZE];
        let agent = Agent {
            position: Position { x: GRID_SIZE / 2, y: GRID_SIZE / 2 },
            health: 100.0,
            oxygen: 100.0,
            levitation: MAX_LEVITATION,
            wands: vec![Wand {
                spells: vec![
                    Spell { id: 1, range: 5, damage: 10.0, cooldown: 3, duration: 1, cost: 5.0, spell_type: "damage".to_string() },
                    Spell { id: 2, range: 8, damage: 0.0, cooldown: 5, duration: 1, cost: 5.0, spell_type: "teleport".to_string() },
                ],
            }],
        };

        // Populate grid with random tiles
        let mut rng = rand::thread_rng();
        for x in 0..GRID_SIZE {
            for y in 0..GRID_SIZE {
                let tile = match rng.gen_range(0..10) {
                    0 => Tile::Solid,
                    1 => Tile::Liquid,
                    2 => Tile::Dangerous,
                    _ => Tile::Empty,
                };
                grid[x][y] = tile;
            }
        }

        Self { grid, agent }
    }

    fn visible_area(&self) -> Vec<Vec<Tile>> {
        let mut visible = vec![vec![Tile::Empty; VISIBLE_AREA]; VISIBLE_AREA];
        let agent_pos = &self.agent.position;

        for i in 0..VISIBLE_AREA {
            for j in 0..VISIBLE_AREA {
                let x = agent_pos.x + i - VISIBLE_AREA / 2;
                let y = agent_pos.y + j - VISIBLE_AREA / 2;
                if x < GRID_SIZE && y < GRID_SIZE {
                    visible[i][j] = self.grid[x][y];
                }
            }
        }
        visible
    }

    fn update_levitation(&mut self, cost: f32) {
        if self.agent.levitation >= cost {
            self.agent.levitation -= cost;
        } else {
            self.agent.levitation = 0.0;
        }
    }
}

// Define the RL agent with a more complex model
#[derive(Debug)]
struct RLAgent {
    conv1: Conv2d,
    conv2: Conv2d,
    lstm: LSTM,
    fc1: Linear,
    fc2: Linear,
}

impl RLAgent {
    fn new(input_shape: Shape) -> Self {
        let conv1 = Conv2d::new(1, 16, (3, 3), (1, 1));
        let conv2 = Conv2d::new(16, 32, (3, 3), (1, 1));
        let lstm = LSTM::new(input_shape, 64);  // LSTM for spell order dependencies
        let fc1 = Linear::new(64, 128);
        let fc2 = Linear::new(128, 3);  // Example output: move, fire wand, use spell

        Self { conv1, conv2, lstm, fc1, fc2 }
    }

    fn forward(&self, input: Tensor<f32>) -> Tensor<f32> {
        let x = input
            .reshape(&[1, 1, VISIBLE_AREA as i64, VISIBLE_AREA as i64])
            .apply(&self.conv1)
            .relu()
            .apply(&self.conv2)
            .relu();
        
        let (lstm_output, _) = self.lstm.forward(x.flatten(2).unsqueeze(0));
        let fc_output = self.fc1.forward(lstm_output.relu()).relu();
        self.fc2.forward(fc_output)
    }
}

// Reward function adjusted for levitation and spell use
fn reward_function(env: &Environment, action: usize, spell_used: Option<&Spell>) -> f32 {
    let mut reward = 0.0;

    // Rewards for reaching objectives or using spells effectively
    if let Some(spell) = spell_used {
        match spell.spell_type.as_str() {
            "damage" => reward += 20.0,
            "heal" => if env.agent.health < 100.0 { reward += 10.0 },
            "teleport" => reward += 5.0,
            _ => {}
        }
    }

    // Running out of oxygen penalty
    if env.agent.oxygen == 0.0 {
        reward -= 5.0;  // Penalty for running out of oxygen
    }

    reward
}

// Training loop with Burn
fn train_agent(env: &mut Environment, agent: &RLAgent, optimizer: &Adam) {
    for _epoch in 0..1000 {
        let state = Tensor::<f32>::zeros(Shape::new([VISIBLE_AREA * VISIBLE_AREA]));

        // Get action and determine spell usage
        let action = agent.forward(state.clone()).argmax().unwrap();
        let spell_used = if action == 1 {
            Some(&env.agent.wands[0].spells[0])  // Example of selecting the first spell
        } else {
            None
        };

        // Calculate the reward based on the action taken
        let reward = reward_function(env, action, spell_used);

        // Update environment (e.g., reduce levitation if agent is moving up)
        if action == 0 {
            env.update_levitation(1.0); // Moving up costs 1.0 levitation
        }

        let loss = -reward;  // Maximize reward

        optimizer.backward_step(loss);
    }
}

fn main() {
    // Initialize environment and complex agent model
    let mut env = Environment::new();
    let agent = RLAgent::new(Shape::new([VISIBLE_AREA * VISIBLE_AREA]));
    let optimizer = Adam::new(1e-4);

    // Training the agent
    train_agent(&mut env, &agent, &optimizer);
}
