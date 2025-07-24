# 🏎️ Webots Driving Game

A thrilling racing game built for the Webots robotics simulator! Drive through checkpoints, collect power-ups, avoid obstacles, and race against AI competitors in a realistic 3D environment.

## 🎮 Game Features

- **🏁 Checkpoint Racing**: Complete laps by driving through all 4 checkpoints
- **⚡ Power-ups**: Collect blue spheres for speed boosts
- **🚧 Obstacles**: Avoid red cylinders to maintain your score
- **🤖 AI Competitor**: Race against an intelligent AI opponent
- **📊 Scoring System**: Earn points for checkpoints, laps, and power-ups
- **🏆 Lap Times**: Track your best lap times
- **🎯 Real-time HUD**: Monitor your progress during the race

## 🎯 Game Objective

Navigate your car around the rectangular race track, passing through all 4 checkpoints in the correct order to complete laps. Collect power-ups for speed boosts while avoiding obstacles. Try to beat your best lap time and compete against the AI!

### Checkpoint Order:
1. 🟢 **Green** - Start/Finish Line
2. 🟡 **Yellow** - Checkpoint 2 (Right side)
3. 🟠 **Orange** - Checkpoint 3 (Top)
4. 🟣 **Purple** - Checkpoint 4 (Left side)

## 🎮 Controls

| Key | Action |
|-----|--------|
| **W** or **↑ Arrow** | Accelerate |
| **S** or **↓ Arrow** | Brake/Reverse |
| **A** or **← Arrow** | Turn Left |
| **D** or **→ Arrow** | Turn Right |
| **SPACEBAR** | Handbrake |
| **R** | Reset car position |
| **ESC** | Quit game |

## 🔧 Installation & Setup (Windows)

### Prerequisites
- **Webots R2023b or later** (Required)
- **Windows 10/11** (64-bit recommended)

### Step 1: Install Webots

1. **Download Webots**:
   - Go to [https://www.cyberbotics.com/](https://www.cyberbotics.com/)
   - Download **Webots R2023b** or later for Windows
   - Choose the complete installation package

2. **Install Webots**:
   - Run the installer as Administrator
   - Follow the installation wizard
   - Install to default location: `C:\Program Files\Webots`
   - Make sure all components are selected

3. **Verify Installation**:
   - Launch Webots from Start Menu
   - You should see the Webots welcome screen

### Step 2: Setup the Game

1. **Extract Game Files**:
   ```
   Extract the game folder to:
   C:\Users\YourName\Documents\WebotsDrivingGame\
   ```

2. **Verify File Structure**:
   ```
   WebotsDrivingGame/
   ├── worlds/
   │   └── driving_game.wbt         ← Main game world
   ├── controllers/
   │   ├── driving_game_controller.py
   │   └── ai_competitor.py
   ├── start_driving_game.py        ← Run this for instructions
   └── DRIVING_GAME_README.md       ← This file
   ```

## 🚀 How to Run the Game

### Method 1: Quick Instructions
1. Run `start_driving_game.py` to see detailed instructions
2. Follow the on-screen setup guide

### Method 2: Manual Steps

1. **Launch Webots**:
   - Open Webots application
   - Wait for it to fully load

2. **Open the Game World**:
   - Click **File** → **Open World...**
   - Navigate to your game folder
   - Select `worlds/driving_game.wbt`
   - Click **Open**

3. **Start the Simulation**:
   - The world will load showing the race track with two cars
   - Click the **▶️ Play** button (or press **Ctrl+2**)
   - The simulation will start

4. **Begin Racing**:
   - The console will show game instructions
   - Use keyboard controls to drive your car (blue car)
   - The red car is the AI competitor

## 📊 Scoring System

| Action | Points |
|--------|--------|
| Pass through checkpoint | +100 |
| Complete a lap | +500 |
| Collect power-up | +50 |
| Hit obstacle | -10 |

## 🏁 Game Elements

### Cars
- **Your Car**: Blue car (player-controlled)
- **AI Competitor**: Red car (computer-controlled)

### Track Elements
- **🟢 Green Checkpoint**: Start/Finish line
- **🟡 Yellow Checkpoint**: Checkpoint 2
- **🟠 Orange Checkpoint**: Checkpoint 3  
- **🟣 Purple Checkpoint**: Checkpoint 4
- **🔵 Blue Spheres**: Speed boost power-ups
- **🔴 Red Cylinders**: Obstacles to avoid
- **🌳 Trees & Buildings**: Environmental decoration

## 🔧 Troubleshooting

### Common Issues:

**Game doesn't start:**
- Ensure Webots R2023b or later is installed
- Make sure you opened the `.wbt` file in Webots
- Click the Play button to start simulation

**No keyboard response:**
- Click inside the Webots 3D view window
- Make sure the simulation is running (Play button pressed)
- Check that the console shows controller messages

**Cars don't move:**
- Reload the world: **File** → **Reload World**
- Check that both controllers are loaded
- Restart Webots if needed

**Performance issues:**
- Close other applications
- Reduce Webots graphics quality in preferences
- Ensure your graphics drivers are updated

**Controller errors:**
- Check that Python files are in the controllers folder
- Ensure file names match exactly
- Look for error messages in the console

## 🎯 Tips for Better Racing

1. **Learn the Track**: Practice the racing line through corners
2. **Use Power-ups Wisely**: Collect blue spheres on straightaways
3. **Avoid Obstacles**: Red cylinders will cost you points
4. **Smooth Steering**: Gentle inputs work better than sharp turns
5. **Watch the AI**: Learn from the computer opponent's racing line

## 🔄 Restarting the Game

To restart the game:
1. Press **Ctrl+Shift+T** to reset the simulation
2. Or use **File** → **Reload World**
3. Press Play to start again

## 📁 File Descriptions

| File | Description |
|------|-------------|
| `worlds/driving_game.wbt` | Main Webots world file |
| `controllers/driving_game_controller.py` | Player car controller |
| `controllers/ai_competitor.py` | AI car controller |
| `start_driving_game.py` | Game launcher with instructions |
| `DRIVING_GAME_README.md` | This instruction file |

## 🚗 Advanced Features

- **Camera Follow**: The camera automatically follows your car
- **Real-time Scoring**: Points are calculated and displayed live
- **Lap Timing**: Track your lap times and personal bests
- **AI Competition**: Smart AI that follows the racing line
- **Physics Simulation**: Realistic car physics and collisions

## 🏆 Challenge Yourself

- Try to complete a lap in under 60 seconds
- Achieve a score over 1000 points
- Complete 5 laps without hitting obstacles
- Beat the AI competitor in a race

## 📞 Support

If you encounter issues:
1. Check this README thoroughly
2. Verify Webots installation
3. Ensure all game files are present
4. Try reloading the world in Webots

## 🎊 Have Fun!

Enjoy racing in the Webots Driving Game! May the best driver win! 🏁

---
*Built with Webots R2023b • Tested on Windows 10/11*