# 🚀 Scrape and Score

### Neural Network powered tool that predicts the top-40 fantasy football scorers of each position each week using years of custom-scraped player & team data.

---

## 📚 Table of Contents

- [About the Project](#about-the-project)
- [Motivation](#motivation)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

---

## 📖 About the Project

This project consists of multiple workflows that users will want to utilize: a) the ability to scrape data across many years into order to train the position specific Neural Networks, b) the ability to scrape upcoming data for NFL games in order to make predictions, c) the ability to update our scraped data in our database following the completion of the NFL games for a given week, and d) the ability to generate the top-40 predictions of players who will be the highest scorer in fantasy. 

---

## 🎯 Motivation

This project was implemented as a result of wanting to combine something I am passionate about (i.e Fantasy Football) with technologies that I wanted to learn about (i.e Python, PyTorch, etc). The goal of being able to actually utilize this application while submitting my fantasy football line ups week to week was a huge driving factor in the amount of time and effort I put into this project. 

---

## ✨ Features

- Generate top-40 player predictions for each relevant NFL fantasy football positon.
- Scrape years of data to power your own Neural Network
- Scrape upcoming data in order to make relevant predictions for an upcoming NFL game

---

## 🛠 Tech Stack

| Technology     | Description                     |
|----------------|---------------------------------|
| Python         | Programming language utilized   |
| PyTorch        | Deep-learning framework         |
| PostgreSQL     | Database                        |

---

## 🚀 Getting Started

### Prerequisites

Python 3.12 installed
PostgreSQL installed

### Steps
1. Create database for your application to connect to 
2. Set up a .env file (see sample.env) within the **scrape_and_score/db** package with relevant configs
3. Create .venv within base directory (i.e <Base-Dir>/scrape_and_score)
4. Activate your .venv and then run the command **pip install -r requirements.txt**
5. Navigate to your database and execute the SQL within the **scrape_and_score/sql** package 
6. Run the **--historical** workflow for multiple year to scrape & persist necessary training data
7. Run the **--nn --train** workflow to generate your Neural Network models
8. Run the **--nn --predict** workflow to generate your predictions



