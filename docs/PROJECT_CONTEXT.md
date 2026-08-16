# Project Context

This was a three-person University of Birmingham robotics team project investigating autonomous exploration in Webots.

## Team-Level System

The full system combined:

- offline tabular Q-learning
- Webots robot deployment
- LiDAR-based safety
- occupancy-grid mapping
- simulated environment testing
- experimental evaluation

## Keng Xiang Tan's Verified Contributions

- developed the baseline Webots experiment controller
- developed the main Webots experiment/deployment controller used to execute trained Q-tables
- tuned controller and LiDAR safety parameters
- worked on World 1 and World 2 simulation setup
- conducted repeated 300-second simulation trials
- logged coverage and collision data
- evaluated behaviour across the original and more cluttered environments

## Team Components

- The core offline Q-learning implementation was a collaborative/team component and is not claimed here as Keng Xiang Tan's sole implementation.
- Occupancy-grid mapping and A* were team components and are not claimed here as Keng Xiang Tan's sole implementation.
- This curated portfolio copy keeps the verified Webots experimentation and evaluation components while leaving out teammate-authored RL and mapping source from `src/`.
