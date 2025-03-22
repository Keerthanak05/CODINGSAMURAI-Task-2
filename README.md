Name : Keerthana

Company : Coding Samurai

C.ID-CSIP1367

Domain : Artificial Intelligence

Duration : 22 February, 2025 and culminate on 22 March, 2025

Overview of the Project

Project : Autonomous Driving Simulation

![road map](https://github.com/user-attachments/assets/0713b5c2-45ff-4f95-aaaa-c7a0339aa22c)

![simple car](https://github.com/user-attachments/assets/6da4fbe1-22f3-4407-af14-f50beeec1e0b)

![WhatsApp Image 2025-03-22 at 10 27 33 (1)](https://github.com/user-attachments/assets/b5d57bfb-a306-4610-aaa9-369604af5c5e)

![WhatsApp Image 2025-03-22 at 10 27 33](https://github.com/user-attachments/assets/86152d0f-d429-41bb-a744-7188776c31a6)

Autonomous Driving Simulation

This project implements an AI-driven Autonomous Driving Simulation using NEAT (NeuroEvolution of Augmenting Topologies) and Pygame. The simulation trains a car to navigate a predefined track using a neural network that evolves over generations.

🚗 Features

Uses NEAT to evolve neural networks for autonomous driving.

Simulates a self-driving car using Pygame.

Implements radar-based sensors to detect obstacles.

Adjusts the car's steering based on neural network decisions.

Uses a fitness-based evaluation to improve driving performance over time.

📌 Requirements

Ensure you have the following dependencies installed before running the simulation:

bash

pip install pygame neat-python

📂 File Structure

main.py → Core simulation logic (car movement, neural network training).

config.txt → Configuration file for NEAT.

road map.png → Track for the car to drive on.

simple car.png → Sprite image for the car.

🏁 How to Run

Clone the repository:

bash

git clone https://github.com/yourusername/autonomous-driving-simulation.git

cd autonomous-driving-simulation

Run the simulation:

bash

python main.py

⚙️ How It Works

The AI-controlled car starts on the track.

It uses radar-based sensors to detect track boundaries.

The neural network makes decisions based on sensor inputs.

Over generations, the AI evolves to improve its driving skills.

📊 Results & Performance

The fitness function rewards cars that drive longer distances without crashing. Over several generations, the AI learns to navigate the track efficiently.

🚀 Future Enhancements

Implement traffic lights and obstacles.

Add real-world driving physics.

Train using reinforcement learning instead of NEAT.
