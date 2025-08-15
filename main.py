# ==========================================================
# Combined Mission Planning App (Single-File Backend + Frontend)
# ==========================================================

import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import heapq
import os
import time
from typing import List, Optional
import uvicorn

# --- 1. Environment Definition (A simplified version for the backend) ---
class MissionEnvironment:
    """
    A 2D grid world environment for mission planning.
    The agent must navigate from a start to a goal position,
    avoiding static obstacles and dynamic threats.
    """
    def __init__(self, grid_size_x=80, grid_size_y=60, num_obstacles=80, num_threats=5, num_movable_obstacles=3):
        # Updated grid dimensions and number of buildings
        self.GRID_SIZE_X = grid_size_x
        self.GRID_SIZE_Y = grid_size_y
        self.AGENT_AVOIDANCE_DISTANCE = 1 # Reduced threat size
        
        # Create a blank grid
        self.grid = np.zeros((self.GRID_SIZE_Y, self.GRID_SIZE_X), dtype=np.int8)
        
        # Place random buildings (obstacles)
        self.obstacles = set()
        self.obstacles_list = []
        for _ in range(num_obstacles):
            while True:
                x, y = random.randint(0, self.GRID_SIZE_X - 1), random.randint(0, self.GRID_SIZE_Y - 1)
                if (x, y) not in self.obstacles:
                    self.grid[y, x] = 1  # 1 for obstacle
                    self.obstacles.add((x, y))
                    self.obstacles_list.append({'x': x, 'y': y})
                    break
            
        self.start_pos = None
        self.goal_pos = None
        
        # Initialize dynamic elements
        self.agent_pos = None
        self.threats = []
        self.movable_obstacles = []
        self.total_rewards = 0.0
        self.done = False
        self.visited_path = [] # Tracks the agent's path
        
        # Observation window (local patch + goal vector)
        self.WINDOW_SIZE = 9
        self.HALF_WINDOW = self.WINDOW_SIZE // 2
        self.num_threats = num_threats
        self.num_movable_obstacles = num_movable_obstacles

    def create_dynamic_entities(self, count, entity_type):
        """Creates a list of dynamic entities with random positions and velocities."""
        entities = []
        for _ in range(count):
            while True:
                x, y = random.randint(0, self.GRID_SIZE_X - 1), random.randint(0, self.GRID_SIZE_Y - 1)
                if (x, y) not in self.obstacles and (x, y) != self.start_pos and (x, y) != self.goal_pos:
                    break
            vx, vy = random.choice([-1, 1]), random.choice([-1, 1])
            entities.append({'pos': (x, y), 'vel': (vx, vy), 'type': entity_type})
        return entities

    def get_state(self):
        """
        Returns a state representation consisting of a flattened local grid
        and a normalized goal direction vector.
        """
        agent_x, agent_y = self.agent_pos
        
        # Padded grid to handle boundaries gracefully
        padded_grid = np.pad(self.grid.copy(), self.HALF_WINDOW, mode='constant', constant_values=-1)
        
        # Mark agent and threats in padded coordinates
        padded_grid[agent_y + self.HALF_WINDOW, agent_x + self.HALF_WINDOW] = 4
        for threat in self.threats:
            t_x, t_y = threat['pos']
            padded_grid[t_y + self.HALF_WINDOW, t_x + self.HALF_WINDOW] = 3
        
        # Correctly extract a window centered on the agent
        window_y_start = agent_y
        window_y_end = agent_y + self.WINDOW_SIZE
        window_x_start = agent_x
        window_x_end = agent_x + self.WINDOW_SIZE
        
        window = padded_grid[window_y_start:window_y_end, window_x_start:window_x_end]

        # Goal direction vector (normalized)
        goal_dx = self.goal_pos[0] - agent_x
        goal_dy = self.goal_pos[1] - agent_y
        max_dist = np.sqrt(self.GRID_SIZE_X**2 + self.GRID_SIZE_Y**2)
        goal_dx_norm = goal_dx / max_dist
        goal_dy_norm = goal_dy / max_dist

        return np.concatenate((window.flatten(), [goal_dx_norm, goal_dy_norm])).astype(np.float32)

    def _manhattan_dist(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self, action):
        """
        Takes an action and returns the new state, reward, and done flag.
        """
        actions = [(0, -1), (0, 1), (-1, 0), (1, 0),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
        dx, dy = actions[action]
        proposed = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

        reward = -0.05
        self.done = False

        dist_before = np.hypot(self.goal_pos[0] - self.agent_pos[0],
                               self.goal_pos[1] - self.agent_pos[1])

        # Check bounds and obstacles
        if not (0 <= proposed[0] < self.GRID_SIZE_X and 0 <= proposed[1] < self.GRID_SIZE_Y):
            reward -= 1.0
            next_pos = self.agent_pos
        elif proposed in self.obstacles:
            reward -= 1.0
            next_pos = self.agent_pos
        else:
            next_pos = proposed

        if not self.done:
            self.agent_pos = next_pos
            self.visited_path.append(self.agent_pos)

            # Goal check
            if self.agent_pos == self.goal_pos:
                reward += 10.0
                self.done = True
            else:
                dist_after = np.hypot(self.goal_pos[0] - self.agent_pos[0],
                                        self.goal_pos[1] - self.agent_pos[1])
                distance_change = dist_before - dist_after
                reward += 0.2 * distance_change

                if distance_change > 0:
                    reward += 0.2 * distance_change
                    self.no_progress_steps = 0
                else:
                    reward -= 0.2
                    self.no_progress_steps += 1

                if self.no_progress_steps > 15:
                    self.done = True
                    
        self.total_rewards += reward
        self.move_threats()
        self.move_movable_obstacles()
        return self.get_state(), reward, self.done

    def move_threats(self):
        """Moves each threat based on its velocity, bouncing off walls and obstacles."""
        for threat in self.threats:
            x, y = threat['pos']
            vx, vy = threat['vel']
            new_x, new_y = x + vx, y + vy
            
            if new_x < 0 or new_x >= self.GRID_SIZE_X:
                vx *= -1
            if new_y < 0 or new_y >= self.GRID_SIZE_Y:
                vy *= -1
            
            new_x, new_y = x + vx, y + vy
            if (new_x, new_y) in self.obstacles:
                vx *= -1
                vy *= -1
                new_x, new_y = x + vx, y + vy
            
            threat['pos'] = (new_x, new_y)
            threat['vel'] = (vx, vy)
    
    def move_movable_obstacles(self):
        """Moves each movable obstacle based on its velocity, bouncing off walls and obstacles."""
        for obstacle in self.movable_obstacles:
            x, y = obstacle['pos']
            vx, vy = obstacle['vel']
            new_x, new_y = x + vx, y + vy
            
            if new_x < 0 or new_x >= self.GRID_SIZE_X:
                vx *= -1
            if new_y < 0 or new_y >= self.GRID_SIZE_Y:
                vy *= -1
            
            new_x, new_y = x + vx, y + vy
            if (new_x, new_y) in self.obstacles:
                vx *= -1
                vy *= -1
                new_x, new_y = x + vx, y + vy
            
            obstacle['pos'] = (new_x, new_y)
            obstacle['vel'] = (vx, vy)

    def reset(self, start=None, destination=None):
        """Resets the environment for a new episode."""
        if start and destination:
            self.start_pos = (start.x, start.y)
            self.goal_pos = (destination.x, destination.y)
        else:
            while True:
                self.start_pos = (random.randint(0, self.GRID_SIZE_X - 1), random.randint(0, self.GRID_SIZE_Y - 1))
                if self.start_pos not in self.obstacles:
                    break
            while True:
                self.goal_pos = (random.randint(0, self.GRID_SIZE_X - 1), random.randint(0, self.GRID_SIZE_Y - 1))
                if self.goal_pos not in self.obstacles and self.goal_pos != self.start_pos:
                    break
        
        self.agent_pos = self.start_pos
        self.total_rewards = 0.0
        self.done = False
        self.no_progress_steps = 0
        self.visited_path = [self.start_pos]
        self.threats = self.create_dynamic_entities(self.num_threats, 'threat')
        self.movable_obstacles = self.create_dynamic_entities(self.num_movable_obstacles, 'movable_obstacle')
        return self.get_state()


# --- 2. DQN Model Definition ---
# Reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

def build_dqn_model(input_shape, num_actions):
    """
    Builds a deep neural network for the DQN agent.
    """
    model = Sequential([
        Input(shape=input_shape),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# --- 3. Mission Manager Class ---
class MissionManager:
    """Manages the mission state, environment, and RL model."""
    def __init__(self, grid_size_x=80, grid_size_y=60, model_path='dqn_model.h5'):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        # Updated number of obstacles to 80
        self.env = MissionEnvironment(grid_size_x=grid_size_x, grid_size_y=grid_size_y, num_obstacles=80, num_threats=5, num_movable_obstacles=3)
        
        # Load the pre-trained RL model
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Successfully loaded model from {model_path}")
        else:
            print(f"⚠️ Model not found at {model_path}. Using random policy.")
            self.model = None

        self.state_size = self.env.WINDOW_SIZE * self.env.WINDOW_SIZE + 2
        self.action_size = 8
        self.current_state = None
        self.step_count = 0
        self.fuel = 0
        self.original_path = []
        self.is_completed = False
        self.mission_success = False

    def _get_path(self, start, destination):
        """Finds a path using the A* algorithm."""
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(node):
            x, y = node
            neighbors = []
            moves = [(0, -1), (0, 1), (-1, 0), (1, 0),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size_x and 0 <= ny < self.grid_size_y and (nx, ny) not in self.env.obstacles:
                    neighbors.append((nx, ny))
            return neighbors
            
        start_node = (start.x, start.y)
        dest_node = (destination.x, destination.y)
        
        frontier = []
        heapq.heappush(frontier, (0, start_node))
        came_from = {}
        cost_so_far = {start_node: 0}
        
        while frontier:
            current_cost, current_node = heapq.heappop(frontier)
            
            if current_node == dest_node:
                break
                
            for next_node in get_neighbors(current_node):
                new_cost = cost_so_far[current_node] + 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(dest_node, next_node)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current_node
        
        path = []
        if dest_node in came_from:
            current = dest_node
            while current != start_node:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            path.reverse()
        return path

    def start_mission(self, start, destination, fuel):
        self.env.reset(start, destination)
        self.current_state = self.env.get_state()
        self.step_count = 0
        self.fuel = fuel
        self.is_completed = False
        self.mission_success = False
        self.original_path = self._get_path(start, destination)
        return self._get_api_state()

    def _get_api_state(self):
        """Formats the current environment state into a Pydantic model for API response."""
        vehicle_pos = self.env.agent_pos
        threats_pos = [t['pos'] for t in self.env.threats]
        movable_obstacles_pos = [t['pos'] for t in self.env.movable_obstacles]
        
        # Map tuples to dictionaries for JSON serialization
        # The radius is now hardcoded to 1 for smaller threats
        api_threats = [{'x': t[0], 'y': t[1], 'radius': 1} for t in threats_pos]
        api_movable_obstacles = [{'x': t[0], 'y': t[1], 'radius': 1} for t in movable_obstacles_pos] # Give them a radius for drawing
        api_orig_path = [{'x': p[0], 'y': p[1]} for p in self.original_path]
        api_replanned_path = [{'x': p[0], 'y': p[1]} for p in self.env.visited_path]
        
        return MissionState(
            vehicle_position={'x': vehicle_pos[0], 'y': vehicle_pos[1]},
            start={'x': self.env.start_pos[0], 'y': self.env.start_pos[1]},
            destination={'x': self.env.goal_pos[0], 'y': self.env.goal_pos[1]},
            step_count=self.step_count,
            fuel=self.fuel,
            is_completed=self.is_completed,
            mission_success=self.mission_success,
            static_obstacles=self.env.obstacles_list, # Send building locations to frontend
            threats=api_threats,
            movable_obstacles=api_movable_obstacles,
            original_path=api_orig_path,
            replanned_path=api_replanned_path
        )

    def next_step(self, new_threat: Optional[dict] = None):
        if self.is_completed:
            return self._get_api_state()

        if new_threat:
            self.env.threats.append({'pos': (new_threat['x'], new_threat['y']), 'vel': (random.choice([-1, 1]), random.choice([-1, 1]))})

        # Use the RL model to predict the next action
        if self.model:
            state_input = self.current_state.reshape(1, self.state_size)
            q_values = self.model.predict(state_input, verbose=0)
            action = np.argmax(q_values[0])
        else:
            action = random.randrange(self.action_size)

        next_state, reward, done = self.env.step(action)
        self.current_state = next_state
        self.step_count += 1
        self.fuel -= 1
        
        if done or self.fuel <= 0:
            self.is_completed = True
            if self.env.agent_pos == self.env.goal_pos:
                self.mission_success = True

        return self._get_api_state()

    def reset(self):
        self.current_state = None
        self.step_count = 0
        self.fuel = 0
        self.is_completed = False
        self.mission_success = False


# --- 4. FastAPI Setup ---
app = FastAPI(title="Mission Route Replanner API")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models for API Requests and Responses
class Point(BaseModel):
    x: int
    y: int

class StartMissionRequest(BaseModel):
    start: Point
    destination: Point
    fuel: int

class Threat(BaseModel):
    x: int
    y: int
    radius: int = 1 # Threat radius reduced to 1

class MovableObstacle(BaseModel):
    x: int
    y: int
    radius: int = 1

class NextStepRequest(BaseModel):
    new_threat: Optional[Threat] = None

class StaticObstacle(BaseModel):
    x: int
    y: int

class MissionState(BaseModel):
    vehicle_position: Optional[Point]
    start: Optional[Point]
    destination: Optional[Point]
    step_count: int
    fuel: int
    is_completed: bool
    mission_success: bool
    static_obstacles: List[StaticObstacle] # Added static obstacles for drawing
    threats: List[Threat]
    movable_obstacles: List[MovableObstacle] 
    original_path: List[Point]
    replanned_path: List[Point]

# Global manager instance
mission_manager = MissionManager(grid_size_x=80, grid_size_y=60) # Updated grid size

@app.post("/start_mission", response_model=MissionState)
async def start_mission_api(request: StartMissionRequest):
    """
    Initializes a new mission with a given start, destination, and fuel.
    """
    return mission_manager.start_mission(request.start, request.destination, request.fuel)

@app.post("/next_step", response_model=MissionState)
async def next_step_api(request: NextStepRequest):
    """
    Advances the mission by one step using the RL agent's policy.
    Can optionally add a new threat to the environment.
    """
    return mission_manager.next_step(request.new_threat)

@app.post("/reset", response_model=dict)
async def reset_api():
    """
    Resets the current mission state.
    """
    mission_manager.reset()
    return {"status": "success", "message": "Mission reset."}

# --- HTML Frontend as a single string ---
HTML_FRONTEND = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Mission Route Replanner</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background: #0b1220; color: #e6eef8; font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }
    .card { background: #0f1724; border: 1px solid #263147; border-radius: 12px; padding: 18px; box-shadow: 0 8px 24px rgba(2,6,23,0.6); }
    canvas { background:#000; border-radius:8px; display:block; width:100%; height:auto; }
    .log p { margin:6px 0; font-size:13px; color:#bcd3ff; }
  </style>
</head>
<body class="p-6">
  <div class="max-w-6xl mx-auto">
    <header class="mb-6 text-center">
      <h1 class="text-3xl font-bold text-sky-400">Mission Route Replanning — Demo</h1>
      <p class="text-sm text-gray-300 mt-1">Frontend (HTML/CSS/JS) + FastAPI backend (RL or fallback random policy)</p>
    </header>

    <div class="grid md:grid-cols-3 gap-6">
      <div class="md:col-span-2 card">
        <canvas id="mapCanvas" width="800" height="600"></canvas>
      </div>

      <div class="card">
        <div class="space-y-3">
          <div class="flex gap-2">
            <button id="startBtn" class="flex-1 bg-sky-500 hover:bg-sky-600 py-2 rounded text-white">Start Mission</button>
            <button id="threatBtn" class="flex-1 bg-amber-500 hover:bg-amber-600 py-2 rounded text-white" disabled>Simulate Threat</button>
          </div>

          <div class="flex gap-2">
            <button id="stepBtn" class="flex-1 bg-emerald-500 hover:bg-emerald-600 py-2 rounded text-white" disabled>Next Step</button>
            <button id="resetBtn" class="flex-1 bg-gray-600 hover:bg-gray-700 py-2 rounded text-white" disabled>Reset</button>
          </div>

          <div>
            <p class="text-sm text-gray-300">Status: <span id="statusText" class="font-medium">Idle</span></p>
            <p class="text-sm text-gray-300">Step: <span id="stepCount">0</span></p>
            <p class="text-sm text-gray-300">Position: <span id="posText">(0,0)</span></p>
            <p class="text-sm text-gray-300">Fuel: <span id="fuelText">N/A</span></p>
          </div>

          <div>
            <h3 class="text-sm text-gray-300 mb-1">Mission Log</h3>
            <div id="log" class="log h-40 overflow-auto p-3 bg-[#071226] border border-[#233047] rounded text-xs"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
(() => {
  const canvas = document.getElementById('mapCanvas');
  const ctx = canvas.getContext('2d');
  const startBtn = document.getElementById('startBtn');
  const threatBtn = document.getElementById('threatBtn');
  const stepBtn = document.getElementById('stepBtn');
  const resetBtn = document.getElementById('resetBtn');
  const statusText = document.getElementById('statusText');
  const stepCountEl = document.getElementById('stepCount');
  const posText = document.getElementById('posText');
  const fuelText = document.getElementById('fuelText');
  const logEl = document.getElementById('log');

  const MAP_W = 800, MAP_H = 600, CELL = 10; // Updated cell size to fit the new grid
  let state = null; // mirror of backend state
  let running = false;

  function log(msg, cls='') {
    const p = document.createElement('p'); p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logEl.prepend(p);
  }

  function draw() {
    ctx.clearRect(0,0,MAP_W,MAP_H);
    // background
    ctx.fillStyle = '#071427'; ctx.fillRect(0,0,MAP_W,MAP_H);
    // grid
    ctx.strokeStyle = '#0f2130'; ctx.lineWidth = 1;
    for (let x=0; x<=MAP_W; x+=CELL) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,MAP_H); ctx.stroke(); }
    for (let y=0; y<=MAP_H; y+=CELL) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(MAP_W,y); ctx.stroke(); }

    if (!state) return;
    // static buildings (obstacles) - New drawing logic
    (state.static_obstacles || []).forEach(b => {
      ctx.fillStyle = '#1c2636';
      ctx.fillRect(b.x*CELL, b.y*CELL, CELL, CELL);
    });
    // original path
    if (state.original_path && state.original_path.length>1) {
      ctx.strokeStyle = '#4f46e5'; ctx.lineWidth = 2;
      ctx.beginPath();
      const p0 = state.original_path[0];
      ctx.moveTo(p0.x*CELL+CELL/2, p0.y*CELL+CELL/2);
      state.original_path.forEach(p => ctx.lineTo(p.x*CELL+CELL/2, p.y*CELL+CELL/2));
      ctx.stroke();
    }
    // replanned
    if (state.replanned_path && state.replanned_path.length>1) {
      ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2;
      ctx.beginPath();
      const p0 = state.replanned_path[0];
      ctx.moveTo(p0.x*CELL+CELL/2, p0.y*CELL+CELL/2);
      state.replanned_path.forEach(p => ctx.lineTo(p.x*CELL+CELL/2, p.y*CELL+CELL/2));
      ctx.stroke();
    }
    // threats
    (state.threats || []).forEach(t => {
      ctx.fillStyle = 'rgba(239,68,68,0.35)';
      ctx.beginPath();
      // Radius now reflects the smaller size
      ctx.arc(t.x*CELL+CELL/2, t.y*CELL+CELL/2, (t.radius||1)*CELL, 0, Math.PI*2); 
      ctx.fill();
    });
    // movable obstacles
    (state.movable_obstacles || []).forEach(o => {
        ctx.fillStyle = '#6b7280';
        ctx.fillRect(o.x*CELL, o.y*CELL, CELL, CELL);
    });
    // vehicle
    if (state.vehicle_position) {
      ctx.fillStyle = '#10b981';
      ctx.fillRect(state.vehicle_position.x*CELL, state.vehicle_position.y*CELL, CELL, CELL);
    }
    // start/destination
    if (state.start) {
      ctx.fillStyle = '#3b82f6';
      ctx.fillRect(state.start.x*CELL, state.start.y*CELL, CELL, CELL);
    }
    if (state.destination) {
      ctx.fillStyle = '#a855f7';
      ctx.fillRect(state.destination.x*CELL, state.destination.y*CELL, CELL, CELL);
    }
  }

  // API helpers
  async function postJSON(path, body) {
    const res = await fetch(path, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body || {})
    });
    if (!res.ok) throw new Error('Server returned ' + res.status);
    return res.json();
  }

  // controls
  startBtn.addEventListener('click', async () => {
    try {
      const start = { x: Math.floor(Math.random() * (MAP_W/CELL)), y: Math.floor(Math.random() * (MAP_H/CELL)) };
      const dest = { x: Math.floor(Math.random() * (MAP_W/CELL)), y: Math.floor(Math.random() * (MAP_H/CELL)) };
      statusText.textContent = 'Starting...';
      const data = await postJSON('/start_mission', { start, destination: dest, fuel: 200 });
      state = data;
      posText.textContent = `(${state.vehicle_position.x}, ${state.vehicle_position.y})`;
      stepCountEl.textContent = state.step_count;
      fuelText.textContent = state.fuel ?? 'N/A';
      statusText.textContent = 'In Progress';
      startBtn.disabled = true; threatBtn.disabled = false; stepBtn.disabled = false; resetBtn.disabled = false;
      log('Mission started: start=' + JSON.stringify(start) + ' dest=' + JSON.stringify(dest));
      draw();
    } catch (err) {
      statusText.textContent = 'Error';
      log('Start failed: ' + err.message);
      console.error(err);
    }
  });

  stepBtn.addEventListener('click', async () => {
    if (!state) return;
    try {
      const data = await postJSON('/next_step', {});
      state = data;
      posText.textContent = state.vehicle_position ? `(${state.vehicle_position.x}, ${state.vehicle_position.y})` : '(n/a)';
      stepCountEl.textContent = state.step_count;
      fuelText.textContent = state.fuel ?? 'N/A';
      if (state.is_completed) {
        statusText.textContent = state.mission_success ? 'Complete' : 'Failed';
        log('Mission ended. success=' + state.mission_success);
        startBtn.disabled = false; threatBtn.disabled = true; stepBtn.disabled = true;
      } else {
        log('Step ' + state.step_count + ' -> pos=' + JSON.stringify(state.vehicle_position));
      }
      draw();
    } catch (err) {
      log('Step error: ' + err.message);
    }
  });

  threatBtn.addEventListener('click', async () => {
    if (!state) return;
    // New threat with reduced radius
    const t = { x: Math.floor(Math.random() * (MAP_W/CELL)), y: Math.floor(Math.random() * (MAP_H/CELL)), radius: 1 }; 
    try {
      const data = await postJSON('/next_step', { new_threat: t });
      state = data;
      log('Threat added at ' + JSON.stringify(t));
      draw();
    } catch (err) {
      log('Threat error: ' + err.message);
    }
  });

  resetBtn.addEventListener('click', async () => {
    try {
      await postJSON('/reset', {});
      state = null;
      statusText.textContent = 'Idle';
      posText.textContent = '(0,0)';
      stepCountEl.textContent = '0';
      fuelText.textContent = 'N/A';
      startBtn.disabled = false; threatBtn.disabled = true; stepBtn.disabled = true; resetBtn.disabled = true;
      log('Mission reset');
      draw();
    } catch (err) {
      log('Reset error: ' + err.message);
    }
  });

  // initial draw
  draw();
})();
</script>
</body>
</html>
"""

@app.get("/")
async def serve_frontend():
    """
    Serves the HTML frontend as a static file.
    """
    return HTMLResponse(content=HTML_FRONTEND, status_code=200)

if __name__ == '__main__':
    # You can now run this single file and access the application at http://127.0.0.1:8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
