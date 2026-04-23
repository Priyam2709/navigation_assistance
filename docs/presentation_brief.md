# Presentation Brief: Navigation Assistance System for Differently-Abled Passengers

## Problem Statement

Railway stations are crowded, visually complex, and stressful for passengers who need mobility support. A passenger using a wheelchair, walker, crutches, or a cane may need help locating destinations such as the platform, restroom, ticket counter, help desk, or exit.

## Project Goal

Build an academic prototype that can:

- detect assistance passengers from station-style video
- track them across multiple frames
- estimate their current station zone
- compute a route to a chosen destination
- deliver readable and audible guidance through a passenger-facing web page

## Proposed Solution

The system takes a video feed, detects passengers with visible mobility aids, and groups repeated detections into tracks. It then maps the passenger's latest zone to a simplified station graph and generates route guidance. The final guidance is exposed through a QR-enabled passenger page with text instructions and browser speech.

## Core Modules

1. Dataset preparation from public mobility-aid and tracking datasets
2. YOLO-based detector training for mobility-aid users
3. Video analysis and passenger tracking
4. Station-zone estimation and route generation
5. Passenger guidance interface with QR and voice output

## Input And Output

Input:

- demo or uploaded station-style video
- trained detector weights
- station graph configuration

Output:

- annotated analysis video
- detected passenger cards
- route summary for a chosen destination
- passenger guidance page
- QR code for mobile access

## Current MVP Strength

- It is not just a model-training project.
- It produces a complete end-to-end workflow from detection to user guidance.
- It includes both an operator-side dashboard and a passenger-side interface.
- It is easy to demonstrate live in class.

## Current Limitations

- The station map is a demo graph, not a calibrated railway station layout.
- The current tracker is lightweight and not a full multi-camera production tracker.
- The public dataset approximates railway behavior but does not replace real station footage.
- The guidance estimate is a demo route estimate, not a field-tested accessibility guarantee.

## Best Demo Flow

1. Open the dashboard.
2. Run `Analyze Bundled Demo Video`.
3. Show the annotated output and the detected assistance passengers.
4. Create a guidance session for one passenger.
5. Open the passenger page.
6. Press `Speak Guidance`.

## One-Line Pitch

This project converts computer-vision detection into practical navigation assistance by linking video understanding, passenger tracking, route planning, and user-facing guidance in one prototype.
