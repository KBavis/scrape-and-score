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

This project was implemented with the intention of being utilzied throughout NFL seasons in order for users to make data-driven / AI decisions when submitting their fantasy football starting line ups. As of now, our application is 

---

## Motivation

This project was implemented as a result of wanting to combine something I am passionate about (i.e Fantasy Football) with technologies that I wanted to learn about (i.e Python, PyTorch, etc). The goal of being able to actually utilize this application while submitting my fantasy football line ups week to week was a huge driving factor in the amount of time and effort I put into this project. 

---

## Features

- Generate top-40 player predictions for each relevant NFL fantasy football positon.
- Scrape years of data to power your own Neural Network
- Scrape upcoming data in order to make relevant predictions for an upcoming NFL game

---

## Tech Stack

| Technology     | Description                     |
|----------------|---------------------------------|
| Python         | Programming language utilized   |
| PyTorch        | Deep-learning framework         |
| PostgreSQL     | Database                        |

---

## Getting Started

### Prerequisites

* Python 3.12 installed locally
* PostgreSQL installed locally

### Steps
1. Create a PostgreSQL Dattabase
2. Set up a .env file (see `sample.env`) within the `scrape_and_score/db` package with relevant configs
3. Create .venv within base directory via the command `python -m venv .venv`
4. Activate your .venv 
    * **macOS/Linux**:
         `source .venv/bin/activate`
    * **Windows (cmd)**:
         `.venv\Scripts\activate.bat`
    * **Windows (PowerShell)**:
         `.venv\Scripts\Activate.ps1`
5. Navigate to your database and execute the SQL within the `scrape_and_score/sql` package 
6. Run the **Historical Workflow** for multiple year to scrape & persist necessary training data
    * `python __main__.py --historical <START_YEAR> <END_YEAR>`
7. Run the **Neural Network Training Workflow** to generate your Neural Network models
    * `python __main__.py --nn --train`
8. Run the **Prediction Workflow** to generate top-40 predictions for each fantasy relevant football position for a specific week/season
    * `python __main__.py --nn --prediction <WEEK> <SEASON>`


## Contributing 

Contributions are welcome! If you'd like to help improve this project, please follow these steps:

1. Clone this repository
    - `git clone https://github.com/KBavis/scrape-and-score.git`
2. Create a new branch:
    - `git checkout -b feat/<YourBranchName>`
3. Commit your changes:
    - `git commit -m "add relevant commit message here!"`
4. Push to branch 
    - `git push origin feature/<YourBranchName>`
5. Open a PR to merge into `main`

**Note**: Please ensure relevant unit tests are included where applicable and the existing style is adhered to. 


## Future Improvements 

Where to even begin! There is so much more I would like to do with this project to optimize its functionality, but for the sake of not writing an essay, I will name the MVP's going forward!

### Neural Network Transfer Learning

One of the big downsides to the current approach of this project is that the amount of data that I am actively using to train our Neural Networks is currently only from 2019 NFL season to current season. The big proponent of this decision is regarding the utilization of **player betting lines** while making our predictions, which I was only able to retrieve as far back as 2019. 

Due to this, I think our models abilities are currently being limited simply by the amount of training data currently available. 

To avoid diminishing the impact of the value of the player betting lines we have available today, I would like to train a base model for a larger number of years (i.e 2000 - 2025) to learn the relationships in other key metrics (i.e weather, age, avg weekly performances, etc) and then create a fine-tuned model that will leverage these pre-trained weights while learning the additional impact of specific player betting lines. 

### Website / API Access
Most users aren't usually going to be command line savvy, making using this application limited to those who do have the necessary knowledge to run this application. To account for this, I would later like to containerize this application utilizing Docker and make this application accessible via a static website, easily allowing for users to skip the data collection portion, and simply leverage the model for it's predictions. 

### Unit Tests
This is a big one. Initially when writing this application, I had actually begun the process of writing unit tests for my newly added functionality. Five refactoring's later, andddd they all ended up being more tech debt then actual help. I plan on adding some essential unit tests (hopefully, if yor reading this, they are already there), but I am sure that this application will still be lacking in that department. 

### Scraping 
A big difficulty I ran into while implementing this application was that certain websites like to change their HTML strucutre fairly often, which ends up breaking a lot fo the scraping functionality. To combat this, a future effort to make scraping as configurable as possible should be made in order to esnure the maintentance of this application over time will not be a complete nightmare. 


## Contact 
Please reach out to me via my email, kellenrbavis@gmail.com, or via my LinkedIn, Kellen Bavis!

## License
This project is licensed under the [MIT License](LICENSE)