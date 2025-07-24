# 🚗 Autonomous Driving Simulation - Installation Guide

## 📋 **System Requirements**
- **OS**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **Graphics**: OpenGL 3.3 support

---

## 🔧 **Step 1: Install Webots Simulator**

### **1.1 Download Webots**
1. Visit [https://cyberbotics.com/](https://cyberbotics.com/)
2. Click **"Download"** → **"Webots R2023b"** (or latest)
3. Select your operating system

### **1.2 Install by Operating System**

**🪟 Windows:**
```powershell
# Download the .exe installer and run it
# Default installation path: C:\Program Files\Webots
# During installation, make sure to:
# ✅ Add Webots to system PATH
# ✅ Install Python integration
```

**🍎 macOS:**
```bash
# Download the .dmg file
# Drag Webots.app to Applications folder
# Path: /Applications/Webots.app
```

**🐧 Linux (Ubuntu/Debian):**
```bash
# Option 1: Using .deb package (recommended)
wget https://github.com/cyberbotics/webots/releases/download/R2023b/webots_2023b_amd64.deb
sudo dpkg -i webots_2023b_amd64.deb
sudo apt-get install -f

# Option 2: Using snap
sudo snap install webots

# Option 3: Manual installation
wget https://github.com/cyberbotics/webots/releases/download/R2023b/webots-R2023b-x86-64.tar.bz2
tar -xjf webots-R2023b-x86-64.tar.bz2
sudo mv webots /opt/
echo 'export PATH="/opt/webots:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### **1.3 Verify Webots Installation**
```bash
# Launch Webots
webots

# Or from GUI: Applications → Webots (Linux/macOS) or Start Menu → Webots (Windows)
```

**Test it works:**
1. Open Webots
2. Go to `File` → `Open Sample World`
3. Navigate to `vehicles` → `highway_overtaking.wbt`
4. Click the ▶️ play button - you should see cars driving!

---

## 🐍 **Step 2: Set Up Python Environment**

### **2.1 Check Python Version**
```bash
python3 --version
# Should show Python 3.8+ (e.g., Python 3.9.7)

# If not installed:
# Windows: Download from python.org
# macOS: brew install python3
# Linux: sudo apt install python3 python3-pip
```

### **2.2 Create Virtual Environment**
```bash
# Create virtual environment
python3 -m venv autonomous_driving

# Activate it:
# Windows:
autonomous_driving\Scripts\activate
# macOS/Linux:
source autonomous_driving/bin/activate

# You should see (autonomous_driving) in your terminal prompt
```

### **2.3 Install Dependencies**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install from requirements file (if you have the project)
pip install -r requirements.txt

# OR install manually:
pip install numpy>=1.21.0
pip install opencv-python>=4.5.0
pip install PyYAML>=6.0
pip install matplotlib>=3.5.0
pip install stable-baselines3>=1.6.0
pip install torch>=1.12.0
pip install tensorflow>=2.8.0
pip install gymnasium>=0.26.0
pip install tqdm>=4.64.0
```

---

## 📂 **Step 3: Get the Project**

### **Option A: Download Project Files**
If you received the project as a zip file:
```bash
# Extract the zip file to your desired location
# cd to the project directory
cd autonomous_driving_project
```

### **Option B: Clone from Repository**
```bash
# If available on GitHub:
git clone <repository-url>
cd autonomous_driving_project
```

### **Option C: Manual Setup**
Create the project structure manually using the files provided in this conversation.

---

## 🎮 **Step 4: Test the Installation**

### **4.1 Quick Test (No Webots Required)**
```bash
# Run the standalone demo
python3 simple_demo.py

# You should see:
# 🚗 Autonomous Driving Demo System
# ==================================================
# [Demo runs showing vehicle simulation]
```

### **4.2 Full Setup Test**
```bash
# Run the setup script
python3 setup.py

# This will:
# ✅ Check Python version
# ✅ Install remaining dependencies
# ✅ Create necessary directories
# ✅ Test all components
```

---

## 🚗 **Step 5: Run the Simulation**

### **5.1 Start Webots with Demo World**

**Method 1: From Webots GUI**
1. Open Webots
2. `File` → `Open World`
3. Navigate to your project folder → `worlds` → `lane.wbt`
4. Click **Open**

**Method 2: From Command Line**
```bash
# Navigate to your project directory
cd /path/to/autonomous_driving_project

# Launch Webots with the world file
webots worlds/lane.wbt
```

### **5.2 Run the Autonomous Vehicle**

**Method 1: Automatic (Recommended)**
1. In Webots, click the ▶️ **Play** button
2. The simulation starts automatically
3. You should see the BMW X5 moving autonomously!

**Method 2: Manual Controller**
```bash
# In a separate terminal (keep Webots running)
cd /path/to/autonomous_driving_project
python3 controllers/autonomous_vehicle.py
```

### **5.3 Available Demo Modes**

```bash
# Basic rule-based driving (30 seconds)
python3 demo.py --mode basic --duration 30

# Complete pipeline demonstration
python3 demo.py --mode comprehensive

# Collect expert driving data
python3 demo.py --mode collect --episodes 10

# Train AI models
python3 demo.py --mode train --algorithm PPO

# Evaluate trained models
python3 demo.py --mode evaluate
```

---

## 🌍 **Step 6: Try Different Environments**

### **Available Worlds:**
1. **Lane Following**: `worlds/lane.wbt`
   - Straight road with obstacles
   - Good for testing basic driving

2. **Roundabout**: `worlds/roundabout.wbt`
   - Circular intersection
   - Tests navigation and yielding

3. **Create Custom Worlds**:
   - Open Webots
   - `File` → `New World`
   - Add roads, vehicles, obstacles
   - Save as `.wbt` file

---

## 🐛 **Troubleshooting**

### **Common Issues & Solutions:**

**❌ "webots: command not found"**
```bash
# Add Webots to PATH:
# Windows: Add C:\Program Files\Webots to system PATH
# macOS: export PATH="/Applications/Webots.app:$PATH"
# Linux: export PATH="/opt/webots:$PATH"
```

**❌ "No module named 'numpy'"**
```bash
# Make sure virtual environment is activated
source autonomous_driving/bin/activate  # Linux/macOS
# autonomous_driving\Scripts\activate   # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**❌ "Controller not found"**
```bash
# Make sure you're in the project directory
cd /path/to/autonomous_driving_project

# Check controller files exist
ls controllers/
# Should show: autonomous_vehicle.py, scripted_vehicle.py
```

**❌ Webots crashes or slow performance**
```bash
# Reduce graphics settings in Webots:
# Tools → Preferences → OpenGL → Disable shadows
# View → Optional Rendering → Uncheck unnecessary options
```

**❌ Python import errors**
```bash
# Check Python path
python3 -c "import sys; print(sys.path)"

# Add project directory to Python path
export PYTHONPATH="/path/to/project:$PYTHONPATH"
```

---

## 🎯 **Quick Start Commands**

Once everything is installed:

```bash
# 1. Activate Python environment
source autonomous_driving/bin/activate

# 2. Go to project directory
cd autonomous_driving_project

# 3. Test without Webots
python3 simple_demo.py

# 4. Test with Webots
webots worlds/lane.wbt &
python3 demo.py --mode basic

# 5. Full demonstration
python3 demo.py --mode comprehensive
```

---

## 📚 **Next Steps**

1. **Explore the Code**: Check out `controllers/autonomous_vehicle.py`
2. **Modify Behavior**: Edit `config/environment_config.yaml`
3. **Add Sensors**: Modify the vehicle in Webots
4. **Train Custom Models**: Use your own driving data
5. **Create New Worlds**: Design custom scenarios

---

## 🆘 **Getting Help**

If you encounter issues:
1. Check this troubleshooting section
2. Read `QUICK_START.md` for additional info
3. Run `python3 setup.py` to verify installation
4. Check Webots documentation: [cyberbotics.com/doc](https://cyberbotics.com/doc)

**Happy autonomous driving! 🚗💨**