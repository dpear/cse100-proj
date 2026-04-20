## Raw Data

The `raw` folder should contain the following 8 files:
- 01-Grades_CSE100_2022-4_Fall.xlsx
- 02-Grades_CSE100_2023-1_Winter.xlsx
- 03-Grades_CSE100_2023-4_Fall.xlsx
- 04-Grades_CSE100_2024-1_Winter.xlsx
- 05-Grades_CSE100_2024-2_Spring.xlsx
- 06-Grades_CSE100_2024-4_Fall.xlsx
- 07-Grades_CSE100_2025-1_Winter.xlsx
- 08-Grades_CSE100_2025-4_Fall.xlsx

These contain grade information from the respective course offerings of CSE100 / CSE100R.
Some of the column names are duplicates because when viewed as a spreadsheet, the first occurence of a column contains the raw score and the second occurence contains the grade out of the total number of points the assignment was worth. This information is stored in the first column, so when reading in data, we will skip this first line and when processing columns, we will use the second occurence of a column.

For the paper, only the last two files, or grades from **Winter 2025** and **Fall 2025** are considered because the course structure was slightly different for all other iterations, therefore it's not best practice to compare all quarters.

### Columns

The following columns should be in files 07 and 08:
- `Section`: CSE100R or CSE100 (remote or in person)
- `Preparation`: Preparation category grade
- `Examination`: Examination category grade
- `Application`: Application category grade
- `Overall`: Students overall grade
- `Midterm`: Grade on the midterm
- `Final`: Grade on the final
- Columns that describe reading quiz grades, either numbered, or by date