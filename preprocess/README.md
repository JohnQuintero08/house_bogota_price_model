PREPROCESS OVERVIEW:

- The scraped date was fully formated and cleaned:
  - The ammenities were separated in different columns having new 160 columns.
  - The data types were adjusted as necessary: integeres for price, float for area, dates as datetime
  - The new columns names were asign to facilitate the understanding of the variables.
- These variables are not important for the model:
  - type has only one value
  - status has 880 empty rows
  - private_area and built_area were condensed in one new columns
- Delete the values of the houses that were registered wrong in the platform, that were extremely high.
- The age was mapped and converted to categories defied as numbers, like ordinal encoding.
