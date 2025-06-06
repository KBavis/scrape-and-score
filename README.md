# Scrape and Score

### AI-driven predictions for the top 40 weekly fantasy football scorers at each position — powered by years of custom-scraped data.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Motivation](#motivation)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

---
## About the Project

This project was created to assist users throughout the NFL season by helping them make data-driven, AI-supported decisions when setting their fantasy football lineups. The application is *far* from finished and will continue to be enhanced and maintained with new features and improvements.

As of **June 5, 2025**, neural network testing is producing the following results, measured within a threshold of 5 fantasy points:

- **Running Back (RB) Model**
  - Accuracy: **65%**
  - Loss: **38.89**

- **Wide Receiver (WR) Model**
  - Accuracy: **70%**
  - Loss: **39.02**

- **Tight End (TE) Model**
  - Accuracy: **75%**
  - Loss: **26.79**

- **Quarterback (QB) Model**
  - Accuracy: **50%**
  - Loss: **52.37**


---

## Motivation

This project stemmed from my desire to combine something I'm passionate about (fantasy football) with technologies I wanted to learn (like Python, PyTorch, etc.). The ability to use this application while setting my fantasy lineups each week was a major motivation behind the time and effort put into it.

---

## Features

- Generate top-40 player predictions for each relevant NFL fantasy football position.
- Scrape years of historical data to power your own neural network.
- Scrape upcoming weekly data to make real-time predictions for upcoming NFL games.

---

## Tech Stack

| Technology | Description                |
|------------|----------------------------|
| Python     | Programming language used  |
| PyTorch    | Deep-learning framework    |
| PostgreSQL | Relational database system |

---

## Getting Started

### Prerequisites

- Python 3.12 installed locally
- PostgreSQL installed locally

### Steps

1. Create a PostgreSQL database.
2. Set up a `.env` file (see `sample.env`) within the `scrape_and_score/db` directory with relevant configurations.
3. Create a virtual environment within the base directory:
   - `python -m venv .venv`
4. Activate the virtual environment:
   - **macOS/Linux**: `source .venv/bin/activate`
   - **Windows (cmd)**: `.venv\Scripts\activate.bat`
   - **Windows (PowerShell)**: `.venv\Scripts\Activate.ps1`
5. Navigate to your database and execute the SQL in the `scrape_and_score/sql` directory.
6. Run the **Historical Workflow** to scrape and persist training data:
   - `python __main__.py --historical <START_YEAR> <END_YEAR>`
7. Run the **Neural Network Training Workflow** to generate your models:
   - `python __main__.py --nn --train`
8. Run the **Prediction Workflow** to generate top-40 predictions per position:
   - `python __main__.py --nn --prediction <WEEK> <SEASON>`

---

## Contributing

Contributions are welcome! To contribute:

1. Clone this repository:
   - `git clone https://github.com/KBavis/scrape-and-score.git`
2. Create a new branch:
   - `git checkout -b feat/<YourBranchName>`
3. Commit your changes:
   - `git commit -m "Add relevant commit message here!"`
4. Push to your branch:
   - `git push origin feat/<YourBranchName>`
5. Open a pull request (PR) to merge into `main`

**Note**: Please ensure relevant unit tests are included where applicable and follow the existing code style.

---

## Future Improvements

There’s a lot I want to enhance in this project, but here’s a breakdown of the most important milestones:

### Neural Network Transfer Learning

Currently, training data is limited to the 2019–present seasons because player betting lines were only available starting in 2019. While this feature is important, the limited dataset may constrain model performance.

A future goal is to train a base model on data spanning 2000–2025 using general player/team stats (e.g., weather, age, average weekly performance) and then fine-tune a model from 2019–2025 that incorporates the betting lines.

### Website / API Access

Most users aren’t comfortable with the command line. To increase usability, I plan to containerize the app with Docker and serve it via a static website. This would allow users to skip data collection and directly use the predictive model.

### Unit Tests

Early on, I started writing unit tests, but repeated refactors left them broken and outdated. Rebuilding a strong suite of unit tests is a priority to ensure long-term maintainability.

### Scraping Reliability

Some sites frequently change their HTML structures, which can break scrapers. A future improvement would be to make the scraping configuration more robust and flexible to reduce maintenance overhead.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

Feel free to reach out:

- 📧 Email: [kellenrbavis@gmail.com](mailto:kellenrbavis@gmail.com)
- 💼 LinkedIn: [Kellen Bavis](https://www.linkedin.com/in/kellen-bavis)
