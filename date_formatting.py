
from datetime import date

def date_formate(date_str):
    """
    :param date_str: Date en chaîne de caractère au format MM/JJ/AA
    :return: objet date représentant cette date
    """
    mounth_str = date_str[0] + date_str[1]
    day_str = date_str[3] + date_str[4]
    year_str = "20" + date_str[6] + date_str[7]

    return date(int(year_str), int(mounth_str), int(day_str))