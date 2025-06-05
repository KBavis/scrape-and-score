from config import props
import logging




def extract_day_from_date(date: str):
   """
   Utility function to extract the day from a string date 

   Args:
      date (str): date to extract day from 

   Returns:
      day (str): day corresponding to date 
   """

   split = date.split("-")
   return split[2]


def extract_home_team_from_game_location(game_location: str):
   """
   Utility function to extract whether or not player was at home 

   Args:
      game_location (str): either an @ sign or nothing 

   Returns: 
      bool (str): T or F indicating home_team or not 
   """

   if game_location == "@":
      return "F"
   else:
      return "T"



def get_team_name_by_pfr_acronym(opp: str):
   """
   Utility function to extract team name from PFR Acronym from properties file 

   Args:
      opp (str): acronym to fetch team for 

   Returns:
      team_name (str): team name corresponding to acronym
   """

   teams = props.get_config("nfl.teams", [])

   for team in teams:
       if team.get("pfr_acronym") == opp:
           return team.get("name")
   
   # account for team name changes over year 
   if "OAK" == opp:
      return "Las Vegas Raiders"
   elif "SDG" == opp:
      return "Los Angeles Chargers"

   logging.warning(f"No team found for acronym: '{opp}'")
   return None

