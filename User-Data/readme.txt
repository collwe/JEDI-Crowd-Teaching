This folder is empty to start, but necessary to store the online data for each user. This is because this data is too large to be stored in RAM if there are multiple hits to the website. This is not the database, and only used during the actual session that the user has with Django.

Do not delete this folder. Deleting the individual files is ok, but you should then also flush the database.