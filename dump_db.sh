mysql jedi -u root -p -e "select * from auth_user" | tr '\t' ',' > users.csv
mysql jedi -u root -p -e "select * from jediteacher_userlabels;" | tr '\t' ',' > labels.csv

