!/bin/sh

wget https://archive.ics.uci.edu/static/public/73/mushroom.zip -P "datasets/"
mkdir "datasets/mushroom/"
mkdir -p "datasets/preprocessed_data/"
unzip "datasets/mushroom.zip" -d "datasets/mushroom/"

wget https://archive.ics.uci.edu/static/public/2/adult.zip -P "datasets/"
mkdir "datasets/adults/"
mkdir -p "datasets/preprocessed_data/"
unzip "datasets/adult.zip" -d "datasets/adults/"

wget https://archive.ics.uci.edu/static/public/14/breast+cancer.zip -P "datasets/"
mkdir "datasets/breastcancer/"
mkdir -p "datasets/preprocessed_data/"
unzip "datasets/breast+cancer.zip" -d "datasets/breastcancer/"




pip install papermill
cd  "datasets/"

papermill "mushroom_dataset_reader.ipynb"     "out1.ipynb"
papermill "adults_dataset_reader.ipynb"       "out2.ipynb"
papermill "breastcancer_dataset_reader.ipynb" "out3.ipynb"


