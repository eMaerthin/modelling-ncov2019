from dataclasses import dataclass


@dataclass
class XlsxFile:
    """Data structure representing an Excel file (XLSX format) given by the name of the file and the name of a sheet
    to read data from"""
    file_name: str
    sheet_name: str


age_gender_xlsx = XlsxFile('age_gender.xlsx', 'processed')
families_and_children_xlsx = XlsxFile('families_and_children.xlsx', 'processed')
families_per_household_xlsx = XlsxFile('families_per_household.xlsx', 'Sheet1')
generations_configuration_xlsx = XlsxFile('generations_configuration.xlsx', 'processed')
household_family_structure_xlsx = XlsxFile('household_family_structure.xlsx', 'Sheet1')
household_family_structure_old_xlsx = XlsxFile('household_family_structure_old.xlsx', 'Sheet1')
households_xlsx = XlsxFile('households.xlsx', 'Sheet1')
households_count_xlsx = XlsxFile('households_count.xlsx', 'processed')
households_old_xlsx = XlsxFile('households_old.xlsx', 'Sheet1')

households_by_master_xlsx = XlsxFile('households_by_master.xlsx', 'House_Master')

output_households_interim_xlsx = XlsxFile('households_interim.xlsx', 'Sheet1')
output_households_xlsx = XlsxFile('households.xlsx', 'Sheet1')
output_population_xlsx = XlsxFile('population.xlsx', 'Sheet1')
production_age = XlsxFile('production_age.xlsx', 'Sheet1')

households_headcount_ac_xlsx_raw = XlsxFile('households_headcount_ac.xlsx', 'Tabl3')
households_headcount_ac_xlsx = XlsxFile('households_headcount_ac.xlsx', 'processed')
